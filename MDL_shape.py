import numpy as np
import collections


class MDL_shape:
    def __init__(
            self,
            series_length,
            correction_cost_weight=1.0,
            min_card=256,
            scale=1000,
    ) -> None:
        self.correction_cost_weight = correction_cost_weight
        self.granularity = series_length // 3
        self.max_segmentation = series_length // 3
        self.min_card = min_card
        self.scale = scale
        self.range_list = None

    class Range:
        def __init__(self, start: int, end: int):
            self.start: int = start
            self.end: int = end

        def contains(self, p: int) -> bool:
            return self.start <= p <= self.end

        def contains_range(self, t: "Range") -> bool:
            return self.contains(t.start) and self.contains(t.end)

        def __str__(self) -> str:
            return "{}-{}".format(self.start, self.end)

        __repr__ = __str__

    def generate_shape(self, input_data):
        shape_data = np.zeros((input_data.shape[0], input_data.shape[1]))
        range_list = []
        point_per_segment = input_data.shape[1] // self.granularity
        for i in range(self.granularity):
            range_start = round(point_per_segment * i)
            if i != self.granularity-1:
                range_end = round(point_per_segment * (i + 1)) - 1
            else:
                range_end = input_data.shape[1] - 1
            index_range = self.Range(range_start, range_end)
            range_list.append(index_range)
        self.range_list = range_list

        for i in range(input_data.shape[0]):
            shape_data[i, :] = self._generate_shape_data_one(input_data[i, :])
        return shape_data

    def _generate_shape_data_one(self, input_data):
        range_values_dict = {}
        min_card = self.min_card
        range_list = self.range_list
        max_seg = self.max_segmentation
        correction_cost_weight = self.correction_cost_weight

        for index_range in range_list:
            range_values_dict[index_range] = np.mean(input_data[index_range.start: index_range.end + 1]) * self.scale
        range_values_discrete = self._discretize_cardinality(range_values_dict, min_card)
        number_of_segments = len(range_list)
        original_values = self._ranges_to_values(range_list, range_values_discrete)

        best_total_cost = float('inf')
        best_ranges = self._deep_copy_ranges(range_values_discrete)

        for card in range(min_card, 257):
            range_values_card = self._discretize_cardinality(range_values_discrete, card)
            dim = number_of_segments
            model_cost = (np.log2(card) + np.ceil(np.log2(number_of_segments))) * dim
            model_values = self._ranges_to_values(range_list, range_values_card)

            correction_values = [x - y for x, y in zip(model_values, original_values)]
            correction_cost = np.sum(np.log2(1 + np.absolute(correction_values)))
            total_cost = model_cost + correction_cost * correction_cost_weight

            if total_cost < best_total_cost and dim <= max_seg:
                best_total_cost = total_cost
                best_ranges = self._deep_copy_ranges(range_values_card)

            merged_range = self._deepcopy_range_list(range_list)
            for dim in range(number_of_segments-1, 0, -1):
                self._merge_two_range(merged_range, range_values_card)

                model_cost = (np.log2(card) + np.ceil(np.log2(number_of_segments))) * dim
                model_values = self._ranges_to_values(range_list, range_values_card)
                correction_values = [x - y for x, y in zip(model_values, original_values)]
                correction_cost = np.sum(np.log2(1 + np.absolute(correction_values)))
                total_cost = model_cost + correction_cost * correction_cost_weight

                if total_cost < best_total_cost and dim <= max_seg:
                    best_total_cost = total_cost
                    best_ranges = self._deep_copy_ranges(range_values_card)
        best_ranges = sorted(best_ranges.items(), key=lambda x: x[0].start)
        best_ranges = collections.OrderedDict(best_ranges)
        return np.array(self._range_to_list(best_ranges)) / self.scale

    @staticmethod
    def _discretize_cardinality(range_values, card):
        range_values_discrete = collections.OrderedDict()
        range_max = max(range_values.values())
        range_min = min(range_values.values())
        cut_points = np.linspace(range_min, range_max, card)
        for index_range, value in range_values.items():
            cut_value = cut_points[np.searchsorted(cut_points, value, side='right') - 1]
            discrete_value = np.ceil(cut_value)
            range_values_discrete[index_range] = int(discrete_value)
        return range_values_discrete

    @staticmethod
    def _ranges_to_values(range_list, range_values_discrete):
        values = []
        for index_range in range_list:
            for range_in_dict, value in range_values_discrete.items():
                if range_in_dict.contains_range(index_range):
                    values.append(value)
                    break
        return values

    @staticmethod
    def _deep_copy_ranges(range_values_discrete):
        new_range_values = collections.OrderedDict()
        for index_range, value in range_values_discrete.items():
            new_range_values[index_range] = value
        return new_range_values

    @staticmethod
    def _deepcopy_range_list(range_list):
        new_range_list = []
        for index_range in range_list:
            new_range_list.append(index_range)
        return new_range_list

    def _merge_two_range(self, merged_range, range_values_card):
        merged_error = [float('inf')] * (len(merged_range) - 1)
        for i in range(len(merged_range) - 1):
            left_segment_value = range_values_card[merged_range[i]]
            right_segment_value = range_values_card[merged_range[i + 1]]
            if left_segment_value <= right_segment_value:
                merged_error[i] = abs(left_segment_value - right_segment_value) * (
                        merged_range[i].end - merged_range[i].start)
            else:
                merged_error[i] = abs(left_segment_value - right_segment_value) * (
                        merged_range[i + 1].end - merged_range[i + 1].start)

        min_error_index = merged_error.index(min(merged_error))
        left_segment_value = range_values_card[merged_range[min_error_index]]
        right_segment_value = range_values_card[merged_range[min_error_index + 1]]
        new_range = self.Range(merged_range[min_error_index].start, merged_range[min_error_index + 1].end)

        if left_segment_value <= right_segment_value:
            new_range_value = range_values_card[merged_range[min_error_index + 1]]
        else:
            new_range_value = range_values_card[merged_range[min_error_index]]

        del range_values_card[merged_range[min_error_index]]
        del range_values_card[merged_range[min_error_index + 1]]
        range_values_card[new_range] = new_range_value
        del merged_range[min_error_index + 1]
        merged_range[min_error_index] = new_range

    @staticmethod
    def _range_to_list(range_values):
        data_list = []
        for index_range in range_values:
            value = range_values[index_range]
            time_len = index_range.end - index_range.start + 1
            data_list.extend([value] * time_len)
        return data_list
