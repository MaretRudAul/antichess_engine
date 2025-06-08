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
    
    def __call__(self, progress_remaining: float) -> float:
        """
        Calculate learning rate based on training progress and curriculum phase.
        
        Args:
            progress_remaining: Float from 1.0 (start) to 0.0 (end)
            
        Returns:
            Current learning rate
        """
        # Convert to progress completed for easier phase comparison
        progress = 1.0 - progress_remaining
        
        # Phase 1: Initial linear schedule 
        if progress < self.phase1_end_pct:
            # Rescale progress for this phase
            phase_progress = progress / self.phase1_end_pct
            phase_remaining = 1.0 - phase_progress
            return self.linear_schedule(phase_remaining)
        
        # Phase 2: Continued linear schedule
        elif progress < self.phase2_end_pct:
            # Rescale progress for this phase
            phase_progress = (progress - self.phase1_end_pct) / (self.phase2_end_pct - self.phase1_end_pct)
            phase_remaining = 1.0 - phase_progress
            return self.linear_schedule(phase_remaining)
        
        # Phase 3: Cosine schedule for fine-tuning with self-play
        else:
            # Rescale progress for final phase
            phase_progress = (progress - self.phase2_end_pct) / (1.0 - self.phase2_end_pct)
            phase_remaining = 1.0 - phase_progress
            return self.cosine_schedule(phase_remaining)