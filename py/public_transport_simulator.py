"""
PT - Public Transport
PV - Private Vehicle
"""

import config.constants_pts as cpts


def get_pt_vals():
    dict_pt_vals = {

        "pt_speed": cpts.PT_SPEED,
        "pt_env": cpts.PT_ENV,
        "pt_cost": cpts.PT_COST,
        "pt_wait": cpts.PT_WAIT,
    }

    return dict_pt_vals


def get_pv_vals():
    dict_pv_vals = {

        "pv_speed": cpts.PV_SPEED,
        "pv_env": cpts.PV_ENV,
        "pv_cost": cpts.PV_COST,
        "pv_wait": cpts.PV_WAIT,
    }

    return dict_pv_vals


def measure_environmental(pt_env, pv_env):
    ans = "env_pv"
    if pt_env >= pv_env:
        ans = "env_pt"
    return ans


def measure_economic(pt_cost, pv_cost):
    ans = "economic_pv"
    if pt_cost <= pv_cost:
        ans = "economic_pt"
    return ans


def measure_time(distance, pt_speed, pt_wait, pv_speed, pv_wait):
    pt_time = pt_wait + distance / pt_speed
    pv_time = pv_wait + distance / pv_speed
    ans = "time_pv"
    if pt_time <= pv_time:
        ans = "time_pt"
    return ans


def compare_pt_pv(pt_vals, pv_vals, **kwargs):
    distance = kwargs.get("distance", 1)
    dict_comparison = {
        "time": measure_time(
            distance,
            pt_vals["pt_speed"],
            pt_vals["pt_wait"],
            pv_vals["pv_speed"],
            pv_vals["pv_wait"]
        ),
        "environmental": measure_environmental(
            pt_vals["pt_env"],
            pv_vals["pv_env"]
        ),
        "economic": measure_economic(
            pt_vals["pt_cost"],
            pv_vals["pv_cost"]
        )
    }
    return dict_comparison


def is_pt_better():
    is_pt_better = True
    pt_vals = get_pt_vals()
    pv_vals = get_pv_vals()
    dict_kwargs = {
        "distance": 20
    }

    dict_comparison = compare_pt_pv(pt_vals, pv_vals, **dict_kwargs)
    print(dict_comparison)

    return is_pt_better


def main():
    is_pt_better()


if __name__ == '__main__':
    main()
