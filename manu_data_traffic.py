import numpy as np

traffic_attr_list = ['BlackCar',
 'GrayCar',
 'WhiteCar',
 'RedCar',
 'OrangeCar',
 'YellowCar',
 'GreenCar',
 'BlueCar',
 'PurpleCar',
 'PinkCar',
# [0,9]
 'BlackBicycle',
 'GrayBicycle',
 'WhiteBicycle',
 'RedBicycle',
 'OrangeBicycle',
 'YellowBicycle',
 'GreenBicycle',
 'BlueBicycle',
 'PurpleBicycle',
 'PinkBicycle',
# [10,19]
 'BlackTruck',
 'GrayTruck',
 'WhiteTruck',
 'RedTruck',
 'OrangeTruck',
 'YellowTruck',
 'GreenTruck',
 'BlueTruck',
 'PurpleTruck',
 'PinkTruck',
# [20,29]
 'BlackSmallVehicles',
 'GraySmallVehicles',
 'WhiteSmallVehicles',
 'RedSmallVehicles',
 'OrangeSmallVehicles',
 'YellowSmallVehicles',
 'GreenSmallVehicles',
 'BlueSmallVehicles',
 'PurpleSmallVehicles',
 'PinkSmallVehicles',
# [30,39]
 'BlackutilityVehicle',
 'GrayutilityVehicle',
 'WhiteutilityVehicle',
 'RedutilityVehicle',
 'OrangeutilityVehicle',
 'YellowutilityVehicle',
 'GreenutilityVehicle',
 'BlueutilityVehicle',
 'PurpleutilityVehicle',
 'PinkutilityVehicle',
# [40,49]
 'BlackTractor',
 'GrayTractor',
 'WhiteTractor',
 'RedTractor',
 'OrangeTractor',
 'YellowTractor',
 'GreenTractor',
 'BlueTractor',
 'PurpleTractor',
 'PinkTractor'
# [50,59]
 ]

traffic_attr_keys = [
    'Car',
    'Bicycle',
    'Truck',
    'SmallVehicles',
    'utilityVehicle',
    'Tractor'
]

traffic_split_attr = [0, 10, 20, 30, 40, 50]

part2attr_dict = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5]
}

split_max_num = 0
for k, v in part2attr_dict.items():
    split_max_num = max(split_max_num, len(v))

attr2part_dict = {
    0: np.array([0]),
    1: np.array([1]),
    2: np.array([2]),
    3: np.array([3]),
    4: np.array([4]),
    5: np.array([5])
}

part2attr_np = np.zeros([len(part2attr_dict.keys()), len(attr2part_dict.keys())]) # 6 -> 6

for k, v in part2attr_dict.items():
    for i in v:
        part2attr_np[k, i] = 1

# 6 * 6  的单位阵
print(part2attr_np.size)