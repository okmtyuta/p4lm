class TimerNotStartedException(Exception):
    def __str__(self):
        return "Time not started"


class TimerNotStoppedException(Exception):
    def __str__(self):
        return "Time  not stopped"


class TimerAlreadyStartedException(Exception):
    def __str__(self):
        return "Time already started"


class TimerAlreadyStoppedException(Exception):
    def __str__(self):
        return "Time  already stopped"
