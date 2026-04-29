"""
command_map.py

Maps ASL letter labels to display characters.
Also handles special gestures like SPACE and DELETE.
"""

# All 26 ASL letters + special gestures
ASL_LABELS = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "SPACE", "DELETE"
]

# Letters that require motion (J and Z trace a path)
# These are harder to classify with static frames alone
MOTION_LETTERS = {"J", "Z"}

# Display color for confidence indicator (BGR for OpenCV)
def get_confidence_color(confidence):
    if confidence >= 0.90:
        return (0, 220, 100)    # Green — high confidence
    elif confidence >= 0.75:
        return (0, 165, 255)    # Orange — medium
    else:
        return (0, 0, 220)      # Red — low

def get_all_labels():
    return ASL_LABELS

def is_motion_letter(label):
    return label in MOTION_LETTERS
