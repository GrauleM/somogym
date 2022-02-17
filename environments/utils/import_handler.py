import sys


def import_environment(environment_name):
    if environment_name == "AntipodalGripper":
        import environments.AntipodalGripper
    elif environment_name == "InHandManipulation":
        import environments.InHandManipulation
    elif environment_name == "InHandManipulationVariableNSeg":
        import environments.InHandManipulationVariableNSeg
    elif environment_name == "InHandManipulationInverted":
        import environments.InHandManipulationInverted
    elif environment_name == "PenSpinner":
        import environments.PenSpinner
    elif environment_name == "PenSpinnerFar":
        import environments.PenSpinnerFar
    elif environment_name == "PlanarBlockPushing":
        import environments.PlanarBlockPushing
    elif environment_name == "PlanarReaching":
        import environments.PlanarReaching
    elif environment_name == "PlanarReachingObstacle":
        import environments.PlanarReachingObstacle
    elif environment_name == "SnakeLocomotionContinuous":
        import environments.SnakeLocomotionContinuous
    elif environment_name == "SnakeLocomotionDiscrete":
        import environments.SnakeLocomotionDiscrete
    elif environment_name == "WormCollective":
        import environments.WormCollective
    elif environment_name == "InchwormLocomotion":
        import environments.InchwormLocomotion
    elif environment_name == "PlanarBoxSpinner":
        import environments.PlanarBoxSpinner
    elif environment_name == "EntanglementGrasping":
        import environments.EntanglementGrasping
    elif environment_name == "Morpheus":
        import environments.Morpheus
    elif environment_name == "CantileverBlockedForce":
        import environments.CantileverBlockedForce
    else:
        print(f"CRITICAL ERROR: Invalid environment '{environment_name}' selected.")
        sys.exit(1)
