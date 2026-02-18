from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    """
    Abstract base class for visuomotor policies.
    """
    
    @abstractmethod
    def reset(self):
        """Reset policy state at the start of an episode."""
        pass
        
    @abstractmethod
    def act(self, obs: dict, info: dict = None) -> np.ndarray:
        """
        Compute action given observation.
        
        Args:
            obs: Observation dictionary from the environment
                 (e.g., {'proprioception': ..., 'visual': ...})
            info: Optional info dictionary containing privileged state
                  
        Returns:
            Action array (normalized or raw, depending on environment config)
        """
        pass


class RandomPolicy(Policy):
    """
    Baseline policy that samples random actions from a given action space.
    """
    
    def __init__(self, action_space):
        """
        Args:
            action_space: The gym action space to sample from
        """
        self.action_space = action_space
        
    def reset(self):
        pass
        
    def act(self, obs: dict, info: dict = None) -> np.ndarray:
        return self.action_space.sample()


class SafePolicy(RandomPolicy):
    """
    Heuristic policy that avoids humans using privileged state information.
    
    Behaves like RandomPolicy when safe, but applies repulsive force
    when human is close.
    """
    
    def __init__(self, action_space, safety_threshold: float = 3.5, repulsion_gain: float = 5.0):
        super().__init__(action_space)
        self.safety_threshold = safety_threshold
        self.gain = repulsion_gain
        self.last_human_pos = None

    def reset(self):
        self.last_human_pos = None
        
    def act(self, obs: dict, info: dict = None) -> np.ndarray:
        # Get base random action
        action = super().act(obs, info)
        
        if info is None:
            return action
            
        safety_info = info.get("safety", {})
        if not safety_info:
            return action
            
        # Get positions (if available)
        robot_pos = safety_info.get("robot_pos")
        human_pos = safety_info.get("human_pos")
        
        if robot_pos is None or human_pos is None:
            return action
            
        r_pos = np.array(robot_pos)
        h_pos = np.array(human_pos)
        
        # Calculate distance
        diff = r_pos - h_pos  # Vector pointing AWAY from human
        dist = np.linalg.norm(diff)
        
        if dist < self.safety_threshold:
            # SAFETY VIOLATION IMMINENT - VELOCITY MATCHING
            
            qpos = safety_info.get("qpos")
            if qpos is not None:
                qpos = np.array(qpos)
                
                if qpos.shape == self.action_space.shape:
                    retreat_action = qpos.copy() # Start with current pose (freeze arms)
                    
                    # Compute Human Velocity (since last step)
                    if self.last_human_pos is not None:
                        h_vel = h_pos - self.last_human_pos
                        
                        # Project human velocity onto Robot->Human vector (diff)
                        # diff = R - H
                        # If dot(h_vel, diff) > 0, human moving towards robot.
                        # If dot(h_vel, diff) < 0, human moving away.
                        
                        dot_prod = np.dot(h_vel[:2], diff[:2])
                        
                        if dot_prod > 0:
                            # Human approaching -> Match Velocity
                            # Apply velocity to robot base (indices 0, 1)
                            velocity_gain = 1.5 
                            base_delta = h_vel[:2] * velocity_gain
                            
                            retreat_action[0] += base_delta[0]
                            retreat_action[1] += base_delta[1]
                        else:
                            # Human moving away or static -> Freeze (as requested initially)
                            # Or naive retreat if very close?
                            # Let's freeze to avoid chasing.
                            pass
                        
                    self.last_human_pos = h_pos
                    return retreat_action
                else:
                    pass

        self.last_human_pos = h_pos
        return action
