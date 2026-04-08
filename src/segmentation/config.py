from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThresholdScoreConfig:
    target_fg_ratio: float = 0.15
    edge_ratio_weight: float = 0.8
    noise_count_cap: int = 80
    noise_weight: float = 0.01
    lineish_weight: float = 0.12
    charish_cap: int = 20
    charish_weight: float = 0.02
    lineish_min_width_ratio: float = 0.20
    lineish_max_height_abs: int = 18
    lineish_max_height_ratio: float = 0.03
    charish_min_height_abs: int = 12
    charish_min_height_ratio: float = 0.02
    charish_max_width_abs: int = 24
    charish_max_width_ratio: float = 0.18


@dataclass(frozen=True)
class HorizontalLineMaskConfig:
    min_span_ratio: float = 0.28
    bridge_gap_ratio: float = 0.05
    max_line_height_ratio: float = 0.10


@dataclass(frozen=True)
class ColoredInkMaskConfig:
    sat_percentile: float = 98.5
    sat_threshold_min: int = 55
    sat_threshold_max: int = 110
    spread_threshold_min: int = 20
    spread_threshold_max: int = 48
    relaxed_sat_ratio: float = 0.45
    relaxed_sat_floor: int = 32


@dataclass(frozen=True)
class FocusRowConfig:
    min_height_abs: int = 20
    min_height_ratio: float = 0.01
    row_snap_min: int = 18
    row_snap_ratio: float = 0.80
    min_group_size: int = 5
    min_span_ratio: float = 0.20
    min_best_score: float = 14.0


@dataclass(frozen=True)
class GenericValidationConfig:
    row_snap_min: int = 18
    row_snap_ratio: float = 0.70
    max_rows: int = 2
    min_line_like: int = 2
    line_like_ratio: float = 0.50
    width_std_abs: float = 18.0
    width_std_ratio: float = 2.2
    guide_dom_min_rects: int = 18
    guide_dom_ratio: float = 0.55
    slim_rect_ratio: float = 0.18


@dataclass(frozen=True)
class UploadCandidateScoreConfig:
    count_weight: float = 2.2
    count_cap: float = 18.0
    span_weight: float = 6.0
    span_cap: float = 5.0
    height_weight: float = 30.0
    height_cap: float = 4.0
    row_bonus_base: float = 2.5
    row_bonus_weight: float = 4.0
    fill_weight: float = 4.0
    fill_cap: float = 2.0
    line_like_weight: float = 3.5
    tiny_ratio_weight: float = 22.0
    width_cv_threshold: float = 0.90
    width_cv_weight: float = 4.0
    height_cv_threshold: float = 0.75
    height_cv_weight: float = 4.0
    aspect_floor: float = 0.36
    aspect_weight: float = 18.0
    slim_ratio_threshold: float = 0.20
    slim_ratio_weight: float = 22.0
    border_touch_weight: float = 0.8
    region_frac_threshold: float = 0.45
    region_frac_weight: float = 10.0


@dataclass(frozen=True)
class SegmentationConfig:
    threshold_score: ThresholdScoreConfig = field(default_factory=ThresholdScoreConfig)
    horizontal_line_mask: HorizontalLineMaskConfig = field(default_factory=HorizontalLineMaskConfig)
    colored_ink_mask: ColoredInkMaskConfig = field(default_factory=ColoredInkMaskConfig)
    focus_row: FocusRowConfig = field(default_factory=FocusRowConfig)
    generic_validation: GenericValidationConfig = field(default_factory=GenericValidationConfig)
    upload_candidate: UploadCandidateScoreConfig = field(default_factory=UploadCandidateScoreConfig)


SEGMENTATION_CONFIG = SegmentationConfig()
