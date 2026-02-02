"""
ISO 15066 Power and Force Limiting (PFL) Limits

Annex A biomechanical limits for quasi-static and transient contact
forces by body region. These values represent the maximum permissible
forces for collaborative robot operation.

Reference: ISO/TS 15066:2016, Annex A, Table A.2
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class BodyRegionLimits:
    """Force and pressure limits for a specific body region."""
    
    name: str
    
    # Force limits (Newtons)
    quasi_static_force: float  # Quasi-static (clamping) contact
    transient_force: float     # Transient (impact) contact
    
    # Pressure limit (N/cm²)
    max_pressure: float
    
    # Effective contact area (m²) - for pressure calculation
    contact_area: float
    
    def get_force_limit(self, contact_type: str) -> float:
        """Get force limit based on contact type."""
        if contact_type == 'quasi_static':
            return self.quasi_static_force
        else:  # transient
            return self.transient_force
    
    def check_violation(
        self, 
        force: float, 
        contact_type: str,
        contact_area: Optional[float] = None,
    ) -> tuple[bool, float]:
        """
        Check if force violates PFL limits.
        
        Args:
            force: Contact force in Newtons
            contact_type: 'quasi_static' or 'transient'
            contact_area: Actual contact area (m²), uses default if None
            
        Returns:
            (is_violation, force_ratio) where ratio > 1.0 means violation
        """
        force_limit = self.get_force_limit(contact_type)
        force_ratio = force / force_limit if force_limit > 0 else 0.0
        
        # Also check pressure if area is provided
        if contact_area is not None and contact_area > 0:
            pressure = force / (contact_area * 10000)  # Convert m² to cm²
            pressure_ratio = pressure / self.max_pressure if self.max_pressure > 0 else 0.0
            # Violation if either force or pressure exceeded
            max_ratio = max(force_ratio, pressure_ratio)
            return max_ratio > 1.0, max_ratio
        
        return force_ratio > 1.0, force_ratio


# ISO 15066 Annex A, Table A.2 - Biomechanical limits
# Values from ISO/TS 15066:2016
PFL_LIMITS: Dict[str, BodyRegionLimits] = {
    # Head/Face
    'skull': BodyRegionLimits(
        name='skull',
        quasi_static_force=130,
        transient_force=260,
        max_pressure=130,
        contact_area=0.00050,  # 5 cm²
    ),
    'face': BodyRegionLimits(
        name='face',
        quasi_static_force=65,
        transient_force=130,
        max_pressure=110,
        contact_area=0.00050,
    ),
    'neck': BodyRegionLimits(
        name='neck',
        quasi_static_force=145,
        transient_force=290,
        max_pressure=140,
        contact_area=0.00100,
    ),
    
    # Torso
    'back_shoulders': BodyRegionLimits(
        name='back_shoulders',
        quasi_static_force=210,
        transient_force=420,
        max_pressure=170,
        contact_area=0.00200,
    ),
    'chest': BodyRegionLimits(
        name='chest',
        quasi_static_force=140,
        transient_force=280,
        max_pressure=120,
        contact_area=0.01000,
    ),
    'abdomen': BodyRegionLimits(
        name='abdomen',
        quasi_static_force=110,
        transient_force=220,
        max_pressure=110,
        contact_area=0.01000,
    ),
    'pelvis': BodyRegionLimits(
        name='pelvis',
        quasi_static_force=180,
        transient_force=360,
        max_pressure=180,
        contact_area=0.00500,
    ),
    
    # Arms
    'upper_arm': BodyRegionLimits(
        name='upper_arm',
        quasi_static_force=150,
        transient_force=300,
        max_pressure=190,
        contact_area=0.00100,
    ),
    'forearm': BodyRegionLimits(
        name='forearm',
        quasi_static_force=160,
        transient_force=320,
        max_pressure=180,
        contact_area=0.00100,
    ),
    'hand_palm': BodyRegionLimits(
        name='hand_palm',
        quasi_static_force=140,
        transient_force=280,
        max_pressure=190,
        contact_area=0.00060,
    ),
    'hand_finger': BodyRegionLimits(
        name='hand_finger',
        quasi_static_force=140,
        transient_force=280,
        max_pressure=300,
        contact_area=0.00010,  # 1 cm²
    ),
    
    # Legs
    'thigh': BodyRegionLimits(
        name='thigh',
        quasi_static_force=220,
        transient_force=440,
        max_pressure=200,
        contact_area=0.00200,
    ),
    'shin': BodyRegionLimits(
        name='shin',
        quasi_static_force=220,
        transient_force=440,
        max_pressure=210,
        contact_area=0.00150,
    ),
    'foot': BodyRegionLimits(
        name='foot',
        quasi_static_force=220,
        transient_force=440,
        max_pressure=210,
        contact_area=0.00150,
    ),
}


# Mapping from SMPL-H collision geom names to ISO body regions
# This maps the _col suffix geoms from smplh_human.xml
GEOM_TO_REGION: Dict[str, str] = {
    # Head
    'Head_col': 'skull',
    'Neck_col': 'neck',
    
    # Torso  
    'Spine_col': 'back_shoulders',
    'Chest_col': 'chest',
    'Torso_col': 'abdomen',
    'Pelvis_col': 'pelvis',
    'L_Thorax_col': 'chest',
    'R_Thorax_col': 'chest',
    
    # Left arm
    'L_Shoulder_col': 'upper_arm',
    'L_Elbow_col': 'forearm',
    'L_Wrist_col': 'hand_palm',
    
    # Right arm
    'R_Shoulder_col': 'upper_arm',
    'R_Elbow_col': 'forearm',
    'R_Wrist_col': 'hand_palm',
    
    # Left leg
    'L_Hip_col': 'pelvis',
    'L_Knee_col': 'thigh',
    'L_Ankle_col': 'shin',
    'L_Toe_col': 'foot',
    
    # Right leg
    'R_Hip_col': 'pelvis',
    'R_Knee_col': 'thigh',
    'R_Ankle_col': 'shin',
    'R_Toe_col': 'foot',
}


def get_region_for_geom(geom_name: str) -> Optional[str]:
    """
    Get the ISO body region for a collision geom.
    
    Args:
        geom_name: Name of the collision geom
        
    Returns:
        ISO region name, or None if not a human body part
    """
    return GEOM_TO_REGION.get(geom_name)


def get_limits_for_geom(geom_name: str) -> Optional[BodyRegionLimits]:
    """
    Get PFL limits for a collision geom.
    
    Args:
        geom_name: Name of the collision geom
        
    Returns:
        BodyRegionLimits, or None if not a human body part
    """
    region = get_region_for_geom(geom_name)
    if region is not None:
        return PFL_LIMITS.get(region)
    return None


def get_all_regions() -> list[str]:
    """Get list of all ISO body region names."""
    return list(PFL_LIMITS.keys())
