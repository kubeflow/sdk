from kubeflow.trainer.backends.localprocess.job import LocalJob


def test_logs_without_follow_returns_split_lines():
    job = LocalJob(name="test-job", command=["echo", "hello"])
    job._stdout = "line1\nline2\n"

    result = job.logs(follow=False)

    assert result == ["line1", "line2"]


def test_logs_with_follow_returns_stream_generator():
    job = LocalJob(name="test-job", command=["echo", "hello"])
    job._stdout = "line1\nline2\n"
    job._output_updated.set()

    # Make the generator think the job has already finished,
    # so it can yield existing output and then exit.
    job.is_alive = lambda: False

    result = list(job.logs(follow=True))

    assert result == ["line1\nline2\n"]