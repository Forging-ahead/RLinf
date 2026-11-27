"""ManiSkill tasks with a spatial manipulation goal and a controllable tool pose."""

EVALUATED_POSE_TASKS = frozenset({
    "DrawSVG-v1",
    "DrawTriangle-v1",
    "FMBAssembly1Easy-v1",
    "InsertFlower-v1",
    "LiftPegUpright-v1",
    "PegInsertionSide-v1",
    "PickCube-v1",
    "PickCubeSO100-v1",
    "PickCubeWidowXAI-v1",
    "PickSingleYCB-v1",
    "PlaceSphere-v1",
    "PlugCharger-v1",
    "PokeCube-v1",
    "PullCube-v1",
    "PullCubeTool-v1",
    "PushCube-v1",
    "PushT-v1",
    "PutCarrotOnPlateInScene-v1",
    "PutEggplantInBasketScene-v1",
    "PutSpoonOnTableClothInScene-v1",
    "RollBall-v1",
    "SO100GraspCube-v1",
    "StackCube-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "StackPyramid-v1",
    "TwoRobotPickCube-v1",
    "TwoRobotStackCube-v1",
    "UnitreeG1PlaceAppleInBowl-v1",
    "UnitreeG1TransportBox-v1",
})

UNSCORED_POSE_TASKS = frozenset({
    "ArchitecTHOR_SceneManipulation-v1",
    "Empty-v1",
    "ReplicaCADPrepareGroceriesTrain_SceneManipulation-v1",
    "ReplicaCADPrepareGroceriesVal_SceneManipulation-v1",
    "ReplicaCADSetTableTrain_SceneManipulation-v1",
    "ReplicaCADSetTableVal_SceneManipulation-v1",
    "ReplicaCADTidyHouseTrain_SceneManipulation-v1",
    "ReplicaCADTidyHouseVal_SceneManipulation-v1",
    "ReplicaCAD_SceneManipulation-v1",
    "RoboCasaKitchen-v1",
    "SceneManipulation-v1",
    "TableTopFreeDraw-v1",
})

# Additional native pose tasks that require separately provisioned assets.
CARTESIAN_TASKS = EVALUATED_POSE_TASKS | UNSCORED_POSE_TASKS | {
    "AssemblingKits-v1",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PickClutterYCB-v1",
    "TurnFaucet-v1",
}
