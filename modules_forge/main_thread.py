# This file is the main thread that handles all Gradio calls for major T2I / I2I processing
# Other Gradio calls (e.g. those from Extensions) are not influenced
# By using one single thread to process all major calls, model moving is significantly faster

import threading
import traceback
from collections import deque
from typing import Callable

lock = threading.Lock()
condition = threading.Condition(lock)

last_id: int = 0
waiting_queue: deque["Task"] = deque()
finished_tasks: dict[int, "Task"] = {}
last_exception: Exception = None


class Task:
    def __init__(self, task_id, func, args, kwargs):
        self.task_id: int = task_id
        self.func: Callable = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None

    def work(self):
        global last_exception
        try:
            self.result = self.func(*self.args, **self.kwargs)
            self.exception = None
            last_exception = None
        except Exception as e:
            traceback.print_exc()
            self.exception = e
            last_exception = e


def loop():
    global waiting_queue, finished_tasks

    while True:
        with condition:
            while not waiting_queue:
                condition.wait()

            task = waiting_queue.popleft()

        task.work()

        with condition:
            finished_tasks[task.task_id] = task
            condition.notify_all()


def async_run(func, *args, **kwargs):
    global last_id

    with condition:
        last_id += 1
        task = Task(task_id=last_id, func=func, args=args, kwargs=kwargs)
        waiting_queue.append(task)
        condition.notify()

        return task.task_id


def run_and_wait_result(func, *args, **kwargs):
    task_id = async_run(func, *args, **kwargs)

    with condition:
        while task_id not in finished_tasks:
            condition.wait()

        task = finished_tasks.pop(task_id)
        return task.result
