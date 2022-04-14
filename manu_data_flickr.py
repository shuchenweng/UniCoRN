import numpy as np

land_attr_list = [
    'blackclouds', 'grayclouds', 'whiteclouds', 'redclouds', 'orangeclouds', 'yellowclouds', 'greenclouds', 'blueclouds', 'purpleclouds', 'pinkclouds',  # [0,9]
    'blackflower', 'grayflower', 'whiteflower', 'redflower', 'orangeflower', 'yellowflower', 'greenflower', 'blueflower', 'purpleflower', 'pinkflower',  # [10,19]
    'blackgrass', 'graygrass', 'whitegrass', 'redgrass', 'orangegrass', 'yellowgrass', 'greengrass', 'bluegrass', 'purplegrass', 'pinkgrass', # [20,29]
    'blackhill', 'grayhill', 'whitehill', 'redhill', 'orangehill', 'yellowhill', 'greenhill', 'bluehill', 'purplehill', 'pinkhill', # [30,39]
    'blackmountain', 'graymountain', 'whitemountain', 'redmountain', 'orangemountain', 'yellowmountain', 'greenmountain', 'bluemountain', 'purplemountain', 'pinkmountain', # [40,49]
    'blacksky-other', 'graysky-other', 'whitesky-other', 'redsky-other', 'orangesky-other', 'yellowsky-other', 'greensky-other', 'bluesky-other', 'purplesky-other', 'pinksky-other', # [50,59]
    'blacktree', 'graytree', 'whitetree', 'redtree', 'orangetree', 'yellowtree', 'greentree', 'bluetree', 'purpletree', 'pinktree', # [60,69]
]
land_attr_keys = ['clouds',  'flower',  'grass',
                 'hill', 'mountain', 'sky-other', 'tree'
                 ]

land_split_attr = [0, 10, 20, 30, 40, 50, 60]

part2attr_dict = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6]
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
    5: np.array([5]),
    6: np.array([6])
}

part2attr_np = np.zeros([len(part2attr_dict.keys()), len(attr2part_dict.keys())]) # 7 -> 7

for k, v in part2attr_dict.items():
    for i in v:
        part2attr_np[k, i] = 1

# 7 * 7  的单位阵
print(part2attr_np.size)

