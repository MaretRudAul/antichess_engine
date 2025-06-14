class LinearSchedule:
    """
    Picklable linear learning rate schedule.
    Compatible with Stable-Baselines3 and multiprocessing.
    """
    
    def __init__(self, initial_value: float, final_value: float = 1e-6):
        """
        Initialize linear schedule.
        
        Args:
            initial_value: Starting learning rate
            final_value: Final learning rate
        """
        self.initial_value = initial_value
        self.final_value = final_value
    
    def __call__(self, progress_remaining: float) -> float:
        """
        Calculate learning rate based on training progress.
        
        Args:
            progress_remaining: Float from 1.0 (start) to 0.0 (end)
            
        Returns:
            Current learning rate
        """
        # Linear interpolation from initial to final value
        return self.final_value + (self.initial_value - self.final_value) * progress_remaining

class CosineSchedule:
    """
    Picklable cosine annealing learning rate schedule.
    Compatible with Stable-Baselines3 and multiprocessing.
    """
    
    def __init__(self, initial_value: float, final_value: float = 1e-6):
        """
        Initialize cosine schedule.
        
        Args:
            initial_value: Starting learning rate
            final_value: Final learning rate
        """
        self.initial_value = initial_value
        self.final_value = final_value
    
    def __call__(self, progress_remaining: float) -> float:
        """
        Calculate learning rate based on training progress.
        
        Args:
            progress_remaining: Float from 1.0 (start) to 0.0 (end)
            
        Returns:
            Current learning rate
        """
        import math
        # Cosine decay from initial to final value
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (1 - progress_remaining)))
        return self.final_value + (self.initial_value - self.final_value) * cosine_decay


class CombinedSchedule:
    """
    Combined linear and cosine schedule with phase transition.
    Linear decay is used for the first portion of training,
    then transitions to cosine annealing for fine-tuning.
    """
    
    def __init__(self, initial_value: float, final_value: float = 1e-6, 
                 linear_pct: float = 0.6):
        """
        Initialize combined schedule.
        
        Args:
            initial_value: Starting learning rate
            final_value: Final learning rate
            linear_pct: Percentage of training that uses linear schedule
        """
        self.linear_schedule = LinearSchedule(initial_value, final_value)
        self.cosine_schedule = CosineSchedule(initial_value, final_value)
        self.linear_pct = linear_pct
        
    def __call__(self, progress_remaining: float) -> float:
        """
        Calculate learning rate based on training progress.
        
        Args:
            progress_remaining: Float from 1.0 (start) to 0.0 (end)
            
        Returns:
            Current learning rate
        """
        # Determine which phase we're in
        training_progress = 1.0 - progress_remaining
        
        if training_progress < self.linear_pct:
            # In linear phase: rescale progress_remaining for this phase
            phase_progress_remaining = 1.0 - (training_progress / self.linear_pct)
            return self.linear_schedule(phase_progress_remaining)
        else:
            # In cosine phase: rescale progress_remaining for this phase
            cosine_phase_progress = (training_progress - self.linear_pct) / (1.0 - self.linear_pct)
            phase_progress_remaining = 1.0 - cosine_phase_progress
            return self.cosine_schedule(phase_progress_remaining)
        
class CurriculumAwareSchedule:
    """
    Learning rate schedule that precisely aligns with curriculum phases:
    - Phase 1 (random opponents): Linear schedule
    - Phase 2 (mixed random/heuristic): Linear schedule
    - Phase 3 (includes self-play): Cosine annealing
    - Phase 4 (primarily self-play): Cosine annealing
    """
    
    def __init__(self, initial_value: float, final_value: float = 1e-6, 
                 curriculum_config=None, total_timesteps=None):
        """
        Initialize curriculum-aware schedule.
        
        Args:
            initial_value: Starting learning rate
            final_value: Final learning rate
            curriculum_config: Reference to curriculum configuration
            total_timesteps: Total training steps
        """
        self.initial_value = initial_value
        self.final_value = final_value
        
        # Create individual schedulers for each phase
        self.linear_schedule = LinearSchedule(initial_value, final_value)
        self.cosine_schedule = CosineSchedule(initial_value, final_value)
        
        # Set curriculum config if provided
        if curriculum_config and total_timesteps:
            self.update_curriculum_config(curriculum_config, total_timesteps)
        else:
            # Default to standard transition points if no config provided
            self.phase1_end_pct = 0.3
            self.phase2_end_pct = 0.6
    
    def update_curriculum_config(self, curriculum_config, total_timesteps):
        """Update the curriculum configuration and recalculate phase transitions."""
        # Calculate phase transition points as percentage of training
        self.phase1_end_pct = curriculum_config["phase_1"]["timesteps"] / total_timesteps
        self.phase2_end_pct = self.phase1_end_pct + (curriculum_config["phase_2"]["timesteps"] / total_timesteps)
        self.phase3_end_pct = self.phase2_end_pct + (curriculum_config["phase_3"]["timesteps"] / total_timesteps)
    
    def __call__(self, progress_remaining: float) -> float:
        """
        Calculate learning rate based on training progress and curriculum phase,
        with smooth transitions between phases.
        """
        # Convert to progress completed for easier phase comparison
        progress = 1.0 - progress_remaining
        
        # Define transition window (5% of training)
        transition_window = 0.05
        
        # Phase 1: Initial linear schedule 
        if progress < self.phase1_end_pct - transition_window:
            # Fully in phase 1
            phase_progress = progress / self.phase1_end_pct
            phase_remaining = 1.0 - phase_progress
            return self.linear_schedule(phase_remaining)
        
        # Transition from Phase 1 to Phase 2
        elif progress < self.phase1_end_pct + transition_window:
            # Blend phase 1 and 2 learning rates
            blend_factor = (progress - (self.phase1_end_pct - transition_window)) / (2 * transition_window)
            
            # Calculate each phase's learning rate
            phase1_lr = self.linear_schedule(1.0 - (progress / self.phase1_end_pct))
            
            phase2_progress = (progress - self.phase1_end_pct) / (self.phase2_end_pct - self.phase1_end_pct)
            phase2_lr = self.linear_schedule(1.0 - phase2_progress)
            
            # Blend the two learning rates
            return phase1_lr * (1 - blend_factor) + phase2_lr * blend_factor
        
        # Phase 2: Continued linear schedule
        elif progress < self.phase2_end_pct - transition_window:
            # Rescale progress for this phase
            phase_progress = (progress - self.phase1_end_pct) / (self.phase2_end_pct - self.phase1_end_pct)
            phase_remaining = 1.0 - phase_progress
            return self.linear_schedule(phase_remaining)
        
        # Transition from Phase 2 to Phase 3
        elif progress < self.phase2_end_pct + transition_window:
            # Blend phase 2 and 3 learning rates
            blend_factor = (progress - (self.phase2_end_pct - transition_window)) / (2 * transition_window)
            
            # Calculate each phase's learning rate
            phase2_progress = (progress - self.phase1_end_pct) / (self.phase2_end_pct - self.phase1_end_pct)
            phase2_lr = self.linear_schedule(1.0 - phase2_progress)
            
            phase3_progress = (progress - self.phase2_end_pct) / (1.0 - self.phase2_end_pct)
            phase3_lr = self.cosine_schedule(1.0 - phase3_progress)
            
            # Blend the two learning rates
            return phase2_lr * (1 - blend_factor) + phase3_lr * blend_factor
        
        # Phase 3: Linear schedule (adding support for Phase 4)
        elif progress < self.phase3_end_pct - transition_window:
            # Rescale progress for this phase
            phase_progress = (progress - self.phase2_end_pct) / (self.phase3_end_pct - self.phase2_end_pct)
            phase_remaining = 1.0 - phase_progress
            return self.linear_schedule(phase_remaining)
        
        # Transition from Phase 3 to Phase 4
        elif progress < self.phase3_end_pct + transition_window:
            # Blend phase 3 and 4 learning rates
            blend_factor = (progress - (self.phase3_end_pct - transition_window)) / (2 * transition_window)
            
            # Calculate each phase's learning rate
            phase3_progress = (progress - self.phase2_end_pct) / (self.phase3_end_pct - self.phase2_end_pct)
            phase3_lr = self.linear_schedule(1.0 - phase3_progress)
            
            phase4_progress = (progress - self.phase3_end_pct) / (1.0 - self.phase3_end_pct)
            phase4_lr = self.cosine_schedule(1.0 - phase4_progress)
            
            # Blend the two learning rates
            return phase3_lr * (1 - blend_factor) + phase4_lr * blend_factor
        
        # Phase 4: Cosine schedule for final phase
        else:
            # Rescale progress for final phase
            phase_progress = (progress - self.phase3_end_pct) / (1.0 - self.phase3_end_pct)
            phase_remaining = 1.0 - phase_progress
            return self.cosine_schedule(phase_remaining)