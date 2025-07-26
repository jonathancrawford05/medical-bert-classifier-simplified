"""
Data Module
Synthetic medical data generation for 10-class classification
"""

from .generate_10class import MedicalDataGenerator

__all__ = [
    "MedicalDataGenerator"
]

# Class information
MEDICAL_CLASSES = {
    0: {"name": "Rx", "description": "Medications and prescriptions"},
    1: {"name": "Lx", "description": "Laboratory tests and results"},
    2: {"name": "Dx", "description": "Medical diagnoses"},
    3: {"name": "Px", "description": "Medical procedures"},
    4: {"name": "Sx", "description": "Symptoms and patient complaints"},
    5: {"name": "Other", "description": "Administrative and non-clinical content"},
    6: {"name": "Vitals", "description": "Vital signs and measurements"},
    7: {"name": "FHx", "description": "Family medical history"},
    8: {"name": "SDOH", "description": "Social determinants of health"},
    9: {"name": "Tobacco", "description": "Smoking and tobacco use"}
}

def get_class_info():
    """Get information about the 10 medical text classes"""
    return MEDICAL_CLASSES

def get_class_names():
    """Get list of class names"""
    return [info["name"] for info in MEDICAL_CLASSES.values()]

def get_class_descriptions():
    """Get mapping of class names to descriptions"""
    return {info["name"]: info["description"] for info in MEDICAL_CLASSES.values()}
