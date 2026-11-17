"""Posture analysis of a person working, from a video or a camera.

A pose model draws the person's skeleton in every frame; the code measures
the angles of their joints and scores them with the methods of
occupational risk prevention (REBA first). Nothing here asks a language
model: the RAG only explains the results.

Every step works frame by frame and never looks ahead, so the same code
analyses a recorded video or a live camera: only the source of the frames
changes.
"""
