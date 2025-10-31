# This file is the main thread that handles all gradio calls for major t2i or i2i processing.
# Other gradio calls (like those from extensions) are not influenced.
# By using one single thread to process all major calls, model moving is significantly faster.


import time
import traceback
import threading
import queue


lock = threading.Lock()
last_id = 0
# waiting tasks queue (thread-safe)
wait_queue = queue.Queue()
# active tasks by id
tasks: dict[int, "Task"] = {}
last_exception = None
stop_event = threading.Event()
_main_thread = None


class Task:
    def __init__(self, task_id, func, args, kwargs):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None
        self.done = threading.Event()

    def work(self):
        global last_exception
        try:
            self.result = self.func(*self.args, **self.kwargs)
            self.exception = None
            last_exception = None
        except Exception as e:
            traceback.print_exc()
            print(e)
            self.exception = e
            last_exception = e
        finally:
            # mark the task as finished so waiters can proceed
            try:
                self.done.set()
            except Exception:
                pass


def loop():
    """Main worker loop.

    This loop checks ``stop_event`` and will exit promptly when it is set.
    Use :pyfunc:`start` to run this loop in a daemon thread and :pyfunc:`stop`
    to request a clean shutdown.
    """
    global lock, last_id, wait_queue, tasks, stop_event
    try:
        while not stop_event.is_set():
            try:
                # block until a task is available or timeout so we can check stop_event
                task = wait_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                task.work()
            finally:
                # keep the task in tasks mapping until waiter collects result
                pass
    except Exception:
        traceback.print_exc()
        # ensure the exception is visible to callers
        # last_exception is set by Task.work on per-task exceptions
    finally:
        # no-op cleanup hook; consumers may call stop() to join the thread
        return


def start():
    """Start the main loop in a background daemon thread.

    If a thread is already running this is a no-op and the running thread is
    returned.
    """
    global _main_thread, stop_event
    stop_event.clear()
    if _main_thread is None or not _main_thread.is_alive():
        _main_thread = threading.Thread(target=loop, name="forge-main-thread", daemon=True)
        _main_thread.start()
    return _main_thread


def stop(timeout=None):
    """Request the main loop to stop and optionally join the thread.

    Returns True if the thread is no longer alive after the join, False if
    the thread is still alive (e.g., if a timeout was given and expired).
    """
    global _main_thread, stop_event
    stop_event.set()
    if _main_thread is not None:
        _main_thread.join(timeout)
        return not _main_thread.is_alive()
    return True


def async_run(func, *args, **kwargs):
    global lock, last_id, wait_queue, tasks
    with lock:
        last_id += 1
        new_task = Task(task_id=last_id, func=func, args=args, kwargs=kwargs)
        tasks[new_task.task_id] = new_task
        wait_queue.put(new_task)
    return new_task.task_id


def run_and_wait_result(func, *args, **kwargs):
    global lock, last_id, tasks
    current_id = async_run(func, *args, **kwargs)

    # wait for the task to be registered
    while True:
        with lock:
            task = tasks.get(current_id)
        if task is not None:
            break
        if stop_event.is_set():
            raise RuntimeError("main_thread stopped before task started")
        stop_event.wait(0.01)

    # wait for the task to finish, but remain responsive to stop_event
    while True:
        if task.done.wait(0.1):
            break
        if stop_event.is_set():
            raise RuntimeError("main_thread stopped before task completed")

    # collect result and cleanup
    try:
        if task.exception is not None:
            raise task.exception
        return task.result
    finally:
        with lock:
            tasks.pop(current_id, None)

