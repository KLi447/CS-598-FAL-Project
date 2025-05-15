import threading
import logging

_shutdown_event = threading.Event()

def get_shutdown_event():
    return _shutdown_event

def request_shutdown(signal_name="Signal"):
    if not _shutdown_event.is_set():
        logging.warning(f"{signal_name} received, initiating graceful shutdown...")
        _shutdown_event.set()

def is_shutdown_requested():
    return _shutdown_event.is_set()