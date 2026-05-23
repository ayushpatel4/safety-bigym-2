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


import re

# Mapping from Unitree G1 collision geom names (the `_col` geoms in
# assets/g1_human_body.xml) to ISO 15066 Annex A body regions. The G1 plays
# the coworker role the SMPL-H human used to; PFL force limits are unchanged,
# only the geom->region keys are remapped. Names that carry a trailing index
# (e.g. the four foot spheres `left_ankle_roll_link_col[0-3]`) are normalised
# by get_region_for_geom, so only the base name needs an entry here.
_G1_BASE_REGIONS: Dict[str, str] = {
    # Trunk / head
    'pelvis': 'pelvis',
    'torso': 'chest',
    'torso_logo': 'chest',
    'head': 'skull',
}
# Per-side limbs (left_/right_ prefixes share the same region).
_G1_LIMB_REGIONS: Dict[str, str] = {
    'hip_pitch_link': 'pelvis',
    'hip_roll_link': 'pelvis',
    'hip_yaw_link': 'thigh',
    'knee_link': 'shin',
    'ankle_pitch_link': 'foot',
    'ankle_roll_link': 'foot',
    'shoulder_pitch_link': 'upper_arm',
    'shoulder_roll_link': 'upper_arm',
    'shoulder_yaw_link': 'upper_arm',
    'elbow_link': 'forearm',
    'wrist_roll_link': 'hand_palm',
    'wrist_pitch_link': 'hand_palm',
    'wrist_yaw_link': 'hand_palm',
}

GEOM_TO_REGION: Dict[str, str] = {
    f'{base}_col': region for base, region in _G1_BASE_REGIONS.items()
}
for _side in ('left', 'right'):
    for _base, _region in _G1_LIMB_REGIONS.items():
        GEOM_TO_REGION[f'{_side}_{_base}_col'] = _region


def get_region_for_geom(geom_name: str) -> Optional[str]:
    """
    Get the ISO body region for a collision geom.

    Args:
        geom_name: Name of the collision geom

    Returns:
        ISO region name, or None if not a coworker body part
    """
    region = GEOM_TO_REGION.get(geom_name)
    if region is not None:
        return region
    # Strip a trailing numeric suffix (e.g. `..._col2` -> `..._col`) so the
    # multi-geom links (feet) resolve to their base region.
    base = re.sub(r"\d+$", "", geom_name)
    return GEOM_TO_REGION.get(base)


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
