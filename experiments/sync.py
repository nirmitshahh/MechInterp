"""Sync OOD: all haystack sequences have same state at 1-after-query index."""
# Rewind so x10 is shared. Need label to predict 2-after. Tests if 2-after uses label.
# x10 ~ N(0,I/5), then rewind each system so they all hit that x10
