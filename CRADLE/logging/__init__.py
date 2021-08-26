import functools
import time


def timer(desc="", level=0, unit="m"):
	if level == 0:
		prefix = "======"
		suffix = "\n"
		completed = "COMPLETED"
	elif level == 1:
		prefix = "--"
		suffix = ""
		completed = "Completed"
	else:
		prefix = "*"
		suffix = ""
		completed = "Completed"

	unit = unit.lower()

	if unit == "h":
		timeDivisor = 3600
		printUnit = "hr(s)"
	elif unit == "m":
		timeDivisor = 60
		printUnit = "min(s)"
	elif unit == "s":
		timeDivisor = 1
		printUnit = "sec(s)"
	else:
		raise TypeError(f"Invalid unit {unit} for `time`")

	def _timer(func):
		"""Print the runtime of the decorated function"""
		@functools.wraps(func)
		def wrapper_timer(*args, **kwargs):
			print(f"{prefix}  {desc} ....")
			startTime = time.perf_counter()    # 1
			value = func(*args, **kwargs)
			endTime = time.perf_counter()      # 2
			runTime = endTime - startTime    # 3

			print(f"{prefix}  {completed} {desc} .... : {runTime / timeDivisor} {printUnit}{suffix}")
			return value
		return wrapper_timer
	return _timer
