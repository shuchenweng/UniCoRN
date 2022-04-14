import numpy as np

vip_attr_list = [
    'nohair', 'blackhair', 'whitehair', 'redhair', 'purplehair', 'yellowhair', 'grayhair', 'bluehair', 'greenhair', 'brownhair', #[0,9]
    'noskin', 'blackskin', 'grayskin', 'whiteskin', 'redskin', 'orangeskin', 'yellowskin', 'greenskin', 'cyanskin', 'blueskin', 'purpleskin', #[10,20]
    'noup', 'blackup', 'grayup', 'whiteup', 'redup', 'orangeup', 'yellowup', 'greenup', 'cyanup', 'blueup', 'purpleup', #[21,31]
    'nocoat', 'blackcoat', 'graycoat', 'whitecoat', 'redcoat', 'orangecoat', 'yellowcoat', 'greencoat', 'cyancoat', 'bluecoat', 'purplecoat', #[32,42]
    'nopants', 'blackpants', 'graypants', 'whitepants', 'redpants', 'orangepants', 'yellowpants', 'greenpants', 'cyanpants', 'bluepants', 'purplepants', #[43,53]
    'noskirt', 'blackskirt', 'grayskirt', 'whiteskirt', 'redskirt', 'orangeskirt', 'yellowskirt', 'greenskirt', 'cyanskirt', 'blueskirt', 'purpleskirt', #[54,64]
    'nodress', 'blackdress', 'graydress', 'whitedress', 'reddress', 'orangedress', 'yellowdress', 'greendress', 'cyandress', 'bluedress', 'purpledress', #[65,75]
    'noshoe', 'blackshoe', 'grayshoe', 'whiteshoe', 'redshoe', 'orangeshoe', 'yellowshoe', 'greenshoe', 'cyanshoe', 'blueshoe', 'purpleshoe', #[76,86]
    'noglove', 'blackglove', 'grayglove', 'whiteglove', 'redglove', 'orangeglove', 'yellowglove', 'greenglove', 'cyanglove', 'blueglove', 'purpleglove', #[87,97]
    'nohat', 'blackhat', 'grayhat', 'whitehat', 'redhat', 'orangehat', 'yellowhat', 'greenhat', 'cyanhat', 'bluehat', 'purplehat', #[98,108]
    'nosock', 'blacksock', 'graysock', 'whitesock', 'redsock', 'orangesock', 'yellowsock', 'greensock', 'cyansock', 'bluesock', 'purplesock', #[109,119]
]

vip_attr_keys = ['hair color', 'skin color', 'upper_clothes color', 'coats color', 'pants color',
                 'skirt color', 'dress color', 'shoes color', 'wearing glove', 'hat color', 'socks color']

vip_split_attr = [
    0, 10, 21, 32, 43, 54, 65, 76, 87, 98, 109
]

part2attr_dict = {
    0: [],      # background？
    1: [9],
    2: [0],
    3: [8],
    4: [],
    5: [2],
    6: [6],
    7: [3],
    8: [10],
    9: [4],
    10: [1],
    11: [],
    12: [5],
    13: [1],
    14: [1],
    15: [1],
    16: [1],
    17: [1],
    18: [7],
    19: [7]
}

split_max_num = 0
for k, v in part2attr_dict.items():
    split_max_num = max(split_max_num, len(v))

attr2part_dict = {
    0: np.array([2]),
    1: np.array([10, 13, 14, 15, 16, 17]),
    2: np.array([5]),
    3: np.array([7]),
    4: np.array([9]),
    5: np.array([12]),
    6: np.array([6]),
    7: np.array([18, 19]),
    8: np.array([3]),
    9: np.array([1]),
    10: np.array([8])
}

part2attr_np = np.zeros([len(part2attr_dict.keys()), len(attr2part_dict.keys())])

for k, v in part2attr_dict.items():
    for i in v:
        part2attr_np[k, i] = 1

#
# Hat 10
# Hair 20
# Glove 30
# Sunglasses 40
# Upper-clothes 50
# Dress 60
# Coat 70
# Socks 80
# Pants 90
# tosor-skin 100
# Scarf 110
# Skirt 120
# Face 130
# Left-arm 140
# Right-arm 150
# Left-leg 160
# Right-leg 170
# Left-shoe 180
# Right-shoe 190

