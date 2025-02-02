class Utils:
    @classmethod
    def rational_split[T](cls, target: list[T], ratios: list[float]) -> list[list[T]]:
        if len(ratios) == 0:
            raise Exception

        results: list[list[T]] = []
        position = 0

        target[0:1]

        for i, ratio in enumerate(ratios):
            if i == len(ratios) - 1:
                splitted = target[position:]
                if len(splitted) > 0:
                    results.append(splitted)
            else:
                size = int(len(target) * ratio / sum(ratios))
                splitted = target[position : position + size]  # noqa: E203
                results.append(splitted)

                position = position + size

        return results

    @classmethod
    def to_even_ratios(cls, target_size: int, unit_size: int):
        ratios = [unit_size / target_size] * (target_size // unit_size)
        ratios.append(1 - sum(ratios))

        return ratios

    @classmethod
    def even_split[T](cls, target: list[T], unit_size) -> list[list[T]]:
        ratios = cls.to_even_ratios(target_size=len(target), unit_size=unit_size)
        return cls.rational_split(target=target, ratios=ratios)
