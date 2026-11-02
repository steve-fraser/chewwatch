"""ChewWatch — UniFi Protect dog-chewing watchdog.

Listens for smart-detect "animal" events on your UniFi Protect cameras, samples
a spread of frames around the event, asks a Kimchi vision model whether the dog
is chewing something it shouldn't, and pushes an ntfy notification with the
best frame attached. Every decision is logged for review and tuning.
"""

__version__ = "0.1.0"
