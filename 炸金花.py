import math
#n!
def jc(n):
    result = 1
    while n >1:
        result = result * n
        n = n - 1
    return result

#Cmn
def combos(m,n):
    up = jc(m)
    down = jc(m-n)*jc(n)
    return up/down

#x^n = y,x=?
def xny(n,y):
    logy = math.log10(y)
    x = pow(10,logy/n)
    return x

all_combos = combos(52,3)
# bomb = 0.00235
bomb = combos(4,3)*13 / all_combos
bombs = ['222','333','444','555','666','777','888','999','TTT','JJJ','QQQ','KKK','AAA']

# straight_flush = 0.00217
straight_flush = 12*4 / all_combos
straight_flushs = ['A23s','234s','345s','456s','567s','678s','789s','89Ts','9TJs','TJQs','JQKs','QKAs']

# flush = 0.04959
flush = (combos(13,3)-12)*4 / all_combos
flushs = ['5 high flush','6 high flush','7 high flush','8 high flush','9 high flush','T high flush','J high flush','Q high flush','K high flush','A high flush']

# straight = 0.03258
straight = (4*4*4-4)*12 / all_combos
straights = ['A23o','234o','345o','456o','567o','678o','789o','89To','9TJo','TJQo','JQKo','QKAo']

# pair = 0.16941
pair = combos(4,2)*combos(48,1)*13 / all_combos
pairs = ['22','33','44','55','66','77','88','99','TT','JJ','QQ','KK','AA']

air = 1 - bomb - straight_flush - flush -straight - pair
airs = ['5 high air','6 high air','7 high air','8 high air','9 high air','T high air','J high air','Q high air','K high air','A high air']

#n人局，对抗所有n-1名玩家的全部范围，想要拥有x%的胜率，至少需要什么牌力
def poker(n,equity):
    least_winning_rate_vs_single_player = xny(n-1,equity)
    #空气
    if least_winning_rate_vs_single_player <= air:
        interval = air / 10  # 10种空气
        num_of_intervals = least_winning_rate_vs_single_player / interval
        # print(interval)
        # print(num_of_intervals)
        return airs[math.ceil(num_of_intervals)-1]
    #对子
    elif air < least_winning_rate_vs_single_player < air + pair:
        interval = pair / 13#13种对子
        num_of_intervals = (least_winning_rate_vs_single_player - air) / interval
        return pairs[math.ceil(num_of_intervals)-1]
    #顺子
    elif air + pair < least_winning_rate_vs_single_player < air + pair + straight:
        interval = straight / 12#12种顺子
        num_of_intervals = (least_winning_rate_vs_single_player - air - pair) / interval
        # print(interval)
        # print(num_of_intervals)
        return straights[math.ceil(num_of_intervals)-1]
    #同花
    elif air + pair + straight < least_winning_rate_vs_single_player < air + pair + straight + flush:
        interval = flush / 10#10种金
        num_of_intervals = (least_winning_rate_vs_single_player - air - pair - straight) / interval
        return flushs[math.ceil(num_of_intervals)-1]
    #同花顺
    elif air + pair + straight + flush < least_winning_rate_vs_single_player < air + pair + straight + flush + straight_flush:
        interval = straight_flush / 12#12种金花
        num_of_intervals = (least_winning_rate_vs_single_player - air - pair - straight - flush) / interval
        return straight_flushs[math.ceil(num_of_intervals)-1]
    #炸弹
    else:
        interval = bomb / 13#12种金花
        num_of_intervals = (least_winning_rate_vs_single_player - air - pair - straight - flush - straight_flush) / interval
        return bombs[math.ceil(num_of_intervals)-1]

if __name__ == '__main__':
    print("all_combos:",all_combos)
    print("bomb:",bomb)
    print("straight_flush:",straight_flush)
    print("flush:",flush)
    print("straight:",straight)
    print("pair:",pair)
    equitys = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    length = len(equitys)
    for j in range(length):
        print("equity:",equitys[j])
        for i in range(3,9):
            print(str(i)+"人局:",poker(i,equitys[j]))
        print()