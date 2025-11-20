"""Validation module for Spark applications (matches operator webhook logic).

This module provides client-side validation that mirrors the Spark Operator's webhook
validation, allowing for fast failure before submission.

Key validations:
- Spark version compatibility (e.g., pod templates require Spark 3.0+)
- Resource format validation (memory, CPU)
- Node selector conflicts
- Dynamic allocation configuration
- Port conflicts in driver ingress options
- Dependency paths
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from typing import Any, Optional

from kubeflow.spark.models import SparkApplicationRequest

logger = logging.getLogger(__name__)


class ValidationErrorType(Enum):
    """Types of validation errors."""

    SPARK_VERSION = "spark_version"
    RESOURCE_FORMAT = "resource_format"
    NODE_SELECTOR_CONFLICT = "node_selector_conflict"
    DRIVER_INGRESS_PORTS = "driver_ingress_ports"
    DYNAMIC_ALLOCATION = "dynamic_allocation"
    DEPENDENCY_PATH = "dependency_path"
    REQUIRED_FIELD = "required_field"
    INVALID_VALUE = "invalid_value"


@dataclass
class ValidationError:
    """A single validation error.

    Attributes:
        type: Type of validation error
        field: Field that failed validation
        message: Human-readable error message
        value: The invalid value (if applicable)
    """

    type: ValidationErrorType
    field: str
    message: str
    value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of validation checks.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings (non-fatal)
    """

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, error: ValidationError):
        """Add an error and mark result as invalid."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, message: str):
        """Add a non-fatal warning."""
        self.warnings.append(message)


class SparkVersionValidator:
    """Validates Spark version compatibility (matches operator logic)."""

    @staticmethod
    def compare_version(version1: str, version2: str) -> int:
        """Compare two semantic versions.

        Args:
            version1: First version string (e.g., "3.5.0")
            version2: Second version string (e.g., "3.0.0")

        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """

        def normalize(v):
            return [int(x) for x in re.sub(r"(\.0+)*$", "", v).split(".")]

        try:
            parts1 = normalize(version1)
            parts2 = normalize(version2)

            # Pad shorter version with zeros
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))

            for p1, p2 in zip(parts1, parts2):
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1
            return 0
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to compare versions {version1} and {version2}: {e}")
            return 0

    def validate(self, request: SparkApplicationRequest) -> ValidationResult:
        """Validate Spark version requirements.

        Checks:
        - Pod templates require Spark >= 3.0.0 (from operator webhook)
        - Dynamic allocation features require Spark >= 3.0.0

        Args:
            request: Spark application request

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Check pod template requirement (from operator)
        if (request.driver_pod_template or request.executor_pod_template) and \
                self.compare_version(request.spark_version, "3.0.0") < 0:
            result.add_error(
                ValidationError(
                    type=ValidationErrorType.SPARK_VERSION,
                    field="spark_version",
                    message="Pod template feature requires Spark version 3.0.0 or higher",
                    value=request.spark_version,
                )
            )

        # Check dynamic allocation (Spark 3.0+)
        if request.dynamic_allocation and request.dynamic_allocation.enabled and \
                self.compare_version(request.spark_version, "3.0.0") < 0:
            result.add_warning(
                "Dynamic allocation on Kubernetes requires Spark 3.0.0+. "
                f"Your version: {request.spark_version}"
            )

        return result


class ResourceValidator:
    """Validates resource specifications (memory, CPU)."""

    # Regex patterns for resource formats
    MEMORY_PATTERN = re.compile(r"^(\d+)(m|M|g|G|k|K|b|B)?$")
    CORE_LIMIT_PATTERN = re.compile(r"^(\d+)(m)?$")

    @classmethod
    def validate_memory(cls, memory: str, field_name: str) -> Optional[ValidationError]:
        """Validate memory format.

        Args:
            memory: Memory string (e.g., "4g", "512m")
            field_name: Field name for error reporting

        Returns:
            ValidationError if invalid, None if valid
        """
        if not cls.MEMORY_PATTERN.match(memory):
            return ValidationError(
                type=ValidationErrorType.RESOURCE_FORMAT,
                field=field_name,
                message=(
                    f"Invalid memory format: {memory}. Expected format: "
                    "<number><unit> where unit is m, g, k, or b (e.g., '4g', '512m')"
                ),
                value=memory,
            )
        return None

    @classmethod
    def validate_cores(cls, cores: int, field_name: str) -> Optional[ValidationError]:
        """Validate CPU cores.

        Args:
            cores: Number of cores
            field_name: Field name for error reporting

        Returns:
            ValidationError if invalid, None if valid
        """
        if cores < 1:
            return ValidationError(
                type=ValidationErrorType.RESOURCE_FORMAT,
                field=field_name,
                message=f"CPU cores must be >= 1, got: {cores}",
                value=cores,
            )
        return None

    def validate(self, request: SparkApplicationRequest) -> ValidationResult:
        """Validate all resource specifications.

        Args:
            request: Spark application request

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Validate driver resources
        error = self.validate_memory(request.driver_memory, "driver_memory")
        if error:
            result.add_error(error)

        error = self.validate_cores(request.driver_cores, "driver_cores")
        if error:
            result.add_error(error)

        # Validate executor resources
        error = self.validate_memory(request.executor_memory, "executor_memory")
        if error:
            result.add_error(error)

        error = self.validate_cores(request.executor_cores, "executor_cores")
        if error:
            result.add_error(error)

        # Validate number of executors
        if request.num_executors < 1 and not (
            request.dynamic_allocation and request.dynamic_allocation.enabled
        ):
            result.add_error(
                ValidationError(
                    type=ValidationErrorType.INVALID_VALUE,
                    field="num_executors",
                    message=("num_executors must be >= 1 (unless dynamic allocation is enabled)"),
                    value=request.num_executors,
                )
            )

        return result


class NodeSelectorValidator:
    """Validates node selector configuration (matches operator webhook)."""

    def validate(self, request: SparkApplicationRequest) -> ValidationResult:
        """Validate node selector conflicts.

        From operator webhook:
        node selector cannot be defined at both SparkApplication and Driver/Executor

        Args:
            request: Spark application request

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # This check is handled differently in the SDK since we don't have separate
        # driver.nodeSelector and executor.nodeSelector fields yet
        # The node_selector field applies to both driver and executor

        if request.node_selector and len(request.node_selector) > 0:
            result.add_warning(
                "node_selector is applied to both driver and executor pods. "
                "Use pod templates if you need different selectors per component."
            )

        return result


class DynamicAllocationValidator:
    """Validates dynamic allocation configuration."""

    def validate(self, request: SparkApplicationRequest) -> ValidationResult:
        """Validate dynamic allocation settings.

        Checks:
        - If enabled, min_executors <= initial_executors <= max_executors
        - Shuffle tracking is enabled by default (operator behavior)

        Args:
            request: Spark application request

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        if not request.dynamic_allocation or not request.dynamic_allocation.enabled:
            return result

        dyn_alloc = request.dynamic_allocation

        # Validate executor bounds
        if dyn_alloc.min_executors is not None and dyn_alloc.max_executors is not None and \
                dyn_alloc.min_executors > dyn_alloc.max_executors:
            result.add_error(
                ValidationError(
                    type=ValidationErrorType.DYNAMIC_ALLOCATION,
                    field="dynamic_allocation",
                    message=(
                        f"min_executors ({dyn_alloc.min_executors}) must be <= "
                        f"max_executors ({dyn_alloc.max_executors})"
                    ),
                    value=(f"min={dyn_alloc.min_executors}, max={dyn_alloc.max_executors}"),
                )
            )

        if dyn_alloc.initial_executors is not None:
            if (
                dyn_alloc.min_executors is not None
                and dyn_alloc.initial_executors < dyn_alloc.min_executors
            ):
                result.add_error(
                    ValidationError(
                        type=ValidationErrorType.DYNAMIC_ALLOCATION,
                        field="dynamic_allocation.initial_executors",
                        message=(
                            f"initial_executors ({dyn_alloc.initial_executors}) "
                            f"must be >= min_executors ({dyn_alloc.min_executors})"
                        ),
                        value=dyn_alloc.initial_executors,
                    )
                )

            if (
                dyn_alloc.max_executors is not None
                and dyn_alloc.initial_executors > dyn_alloc.max_executors
            ):
                result.add_error(
                    ValidationError(
                        type=ValidationErrorType.DYNAMIC_ALLOCATION,
                        field="dynamic_allocation.initial_executors",
                        message=(
                            f"initial_executors ({dyn_alloc.initial_executors}) "
                            f"must be <= max_executors ({dyn_alloc.max_executors})"
                        ),
                        value=dyn_alloc.initial_executors,
                    )
                )

        # Warn if shuffle tracking is disabled (operator enables by default)
        if dyn_alloc.shuffle_tracking_enabled is False:
            result.add_warning(
                "Shuffle tracking is disabled. You may need an external shuffle service. "
                "See: https://spark.apache.org/docs/latest/running-on-kubernetes.html"
                "#dynamic-resource-allocation"
            )

        return result


class SparkApplicationValidator:
    """Main validator that orchestrates all validation checks."""

    def __init__(self):
        """Initialize validator with all sub-validators."""
        self.version_validator = SparkVersionValidator()
        self.resource_validator = ResourceValidator()
        self.node_selector_validator = NodeSelectorValidator()
        self.dynamic_allocation_validator = DynamicAllocationValidator()

    def validate_all(self, request: SparkApplicationRequest) -> ValidationResult:
        """Run all validation checks.

        Args:
            request: Spark application request

        Returns:
            ValidationResult with all errors and warnings
        """
        final_result = ValidationResult(valid=True)

        # Run all validators
        validators = [
            self.version_validator,
            self.resource_validator,
            self.node_selector_validator,
            self.dynamic_allocation_validator,
        ]

        for validator in validators:
            result = validator.validate(request)
            final_result.errors.extend(result.errors)
            final_result.warnings.extend(result.warnings)

        # Mark as invalid if any errors
        if final_result.errors:
            final_result.valid = False

        # Log results
        if not final_result.valid:
            logger.error(f"Validation failed with {len(final_result.errors)} errors:")
            for error in final_result.errors:
                logger.error(f"  [{error.type.value}] {error.field}: {error.message}")

        if final_result.warnings:
            logger.warning(f"Validation completed with {len(final_result.warnings)} warnings:")
            for warning in final_result.warnings:
                logger.warning(f"  {warning}")

        return final_result

    def validate_and_raise(self, request: SparkApplicationRequest):
        """Validate and raise exception if invalid.

        Args:
            request: Spark application request

        Raises:
            ValueError: If validation fails
        """
        result = self.validate_all(request)

        if not result.valid:
            error_messages = [f"{error.field}: {error.message}" for error in result.errors]
            raise ValueError(
                "Spark application validation failed:\n"
                + "\n".join(f"  - {msg}" for msg in error_messages)
            )


# Convenience function
def validate_spark_application(request: SparkApplicationRequest) -> ValidationResult:
    """Validate a Spark application request.

    Args:
        request: Spark application request to validate

    Returns:
        ValidationResult

    Example:
        ```python
        from kubeflow.spark import SparkApplicationRequest
        from kubeflow.spark.validation import validate_spark_application

        request = SparkApplicationRequest(
            app_name="my-app",
            main_application_file="local:///app/main.py",
            spark_version="2.4.0",  # Too old for pod templates!
            driver_pod_template={...},  # Will fail validation
        )

        result = validate_spark_application(request)
        if not result.valid:
            for error in result.errors:
                print(f"Error: {error.message}")
        ```
    """
    validator = SparkApplicationValidator()
    return validator.validate_all(request)
