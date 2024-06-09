
def PoseSelect(PoseName):

    if PoseName == 'frontal':
        pose = 0
    elif PoseName == 'profile':
        pose = 1
    else:
        print('Something wrong!')
        exit()

    return pose