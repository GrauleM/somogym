def vary_segment_number(manipulator_def_dict, n_seg_new):
    # careful: the lines below are only functional for manipulators with only one act
    assert (
        manipulator_def_dict["n_act"] == 1
    ), f"number of segments can only be changed for manipulators with only one actuator (for now)."
    assert (
        manipulator_def_dict["actuator_definitions"][0]["planar_flag"] == 0
    ), f"for now, n_segments can only be changed on non-planar actuators for IHM."
    n_def = manipulator_def_dict["actuator_definitions"][0][
        "n_segments"
    ]  # the number of segments given in the definition
    n_def_per_axis = (
        n_def
        if manipulator_def_dict["actuator_definitions"][0]["planar_flag"] == 1
        else n_def / 2.0
    )
    assert n_def % 2 == 0, f"n_def is not an even number"

    n_ax0 = n_def_per_axis
    n_ax1 = n_def_per_axis
    k_ax0 = manipulator_def_dict["actuator_definitions"][0]["joint_definitions"][0][
        "spring_stiffness"
    ]  # stiffness of axis 0
    k_ax1 = manipulator_def_dict["actuator_definitions"][0]["joint_definitions"][1][
        "spring_stiffness"
    ]  # stiffness of axis 1
    mass = manipulator_def_dict["actuator_definitions"][0]["link_definition"]["mass"]

    # todo: math how to change inertial values; change inertial values here
    # todo: experiments how to change limit force; change limit force here
    # todo: for somo, add a check or demo that assures that the inertial values make sense given the mass and link dimensions
    segment_length = manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "dimensions"
    ][2]

    n_def_per_axis_new = (
        n_seg_new
        if manipulator_def_dict["actuator_definitions"][0]["planar_flag"] == 1
        else n_seg_new / 2.0
    )
    n_ax0_new = n_def_per_axis_new
    n_ax1_new = n_def_per_axis_new
    k_ax0_new = k_ax0 / n_ax0 * n_ax0_new
    k_ax1_new = k_ax1 / n_ax1 * n_ax1_new
    mass_new = mass * n_def / n_seg_new
    segment_length_new = segment_length * n_def / n_seg_new

    Ixx = manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "inertial_values"
    ][0]
    Iyy = manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "inertial_values"
    ][3]
    Izz = manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "inertial_values"
    ][5]

    Izz_new = Izz / mass * mass_new  # since Izz is independent of segment length

    if manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "shape_type"
    ] in ["stadium", "box"]:
        x_dim = manipulator_def_dict["actuator_definitions"][0]["link_definition"][
            "dimensions"
        ][0]
        y_dim = manipulator_def_dict["actuator_definitions"][0]["link_definition"][
            "dimensions"
        ][1]
        Ixx_new = (
            Ixx
            / (mass * (y_dim ** 2 + segment_length ** 2))
            * (mass_new * (y_dim ** 2 + segment_length_new ** 2))
        )
        Iyy_new = (
            Iyy
            / (mass * (x_dim ** 2 + segment_length ** 2))
            * (mass_new * (x_dim ** 2 + segment_length_new ** 2))
        )
    elif manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "shape_type"
    ] in ["capsule", "cylinder"]:
        r = manipulator_def_dict["actuator_definitions"][0]["link_definition"][
            "dimensions"
        ][0]
        Ixx_new = (
            Ixx
            / (mass * (3 * r ** 2 + segment_length ** 2))
            * (mass_new * (3 * r ** 2 + segment_length_new ** 2))
        )
        Iyy_new = (
            Iyy
            / (mass * (3 * r ** 2 + segment_length ** 2))
            * (mass_new * (3 * r ** 2 + segment_length_new ** 2))
        )
    else:
        assert False, (
            f"changing the number of segments has not been implemented for shape_type "
            f"{manipulator_def_dict['actuator_definitions'][0]['link_definition']['shape_type']}"
        )

    # assign everything
    manipulator_def_dict["actuator_definitions"][0]["n_segments"] = n_seg_new
    manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "mass"
    ] = mass_new
    manipulator_def_dict["actuator_definitions"][0]["link_definition"]["dimensions"][
        2
    ] = segment_length_new
    manipulator_def_dict["actuator_definitions"][0]["joint_definitions"][0][
        "spring_stiffness"
    ] = k_ax0_new
    manipulator_def_dict["actuator_definitions"][0]["joint_definitions"][1][
        "spring_stiffness"
    ] = k_ax1_new

    manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "inertial_values"
    ][0] = Ixx_new
    manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "inertial_values"
    ][3] = Iyy_new
    manipulator_def_dict["actuator_definitions"][0]["link_definition"][
        "inertial_values"
    ][5] = Izz_new

    return manipulator_def_dict
