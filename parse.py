import numpy as np
import pandas as pd

def convert_draft_str_to_array(x):

	def get_heroes_labels():	
		heroes = pd.read_csv("./resources/dota2_heroes.csv")
		x_label = np.array(heroes['id'])
		# x_label = np.c_[x_label, np.full_like(x_label, 0.5, dtype=float)]
		x_label = np.c_[x_label, np.zeros(len(x_label))]
		return x_label

	def convert_team_to_list(string):
		lst = string.split(',')
		# print (string)
		# print (lst)
		return list(np.float_(lst))

	def prework_data(x, x_label):
		new_x = np.array([[]])
		new_x_index = 0
		for index, row in enumerate(x):
			new_a = np.copy(x_label)
			radiant = 0
			# if index == 1101:
			# 	print (index)
			for team in row:
				# if index == 1101:
				# 	print (team)
				for hero_id in convert_team_to_list(team):
					i = 0
					while i < len(new_a):
						if hero_id == new_a[i][0] and radiant < 5:
							new_a[i][1] = 1
						elif hero_id == new_a[i][0] and radiant >= 5:
							new_a[i][1] = -1
						i += 1
					radiant += 1
			new_a = new_a.transpose()[1]
			if new_x_index == 0:
				new_x = np.append(new_x, [new_a], 1)
			else:
				new_x = np.append(new_x, [new_a], 0)
			new_x_index += 1
		return new_x

	# print (type(x[0][0]))
	# print (x[0])
	x_label = get_heroes_labels()
	new_x = prework_data(x, x_label)

	return x_label, new_x

def	remove_empty_row(data):
	data = data[data.astype(str)['draft'] != '[]']
	return data

def dataframe_to_draft_str(data):
    draft = np.array(data['draft'])
    # match_id = np.array(data['match_id'])

    match_heroes = []
    for matches in draft:
        #active_team : 2 = radiant, 3 = dire
        str_radiant = ""
        str_dire = ""
        j = 0
        k = 0
        rad = 3
        choose = True
        for i in range(len(matches)):
            if matches[i].get('pick') == True:
                if choose == True:
                    if matches[i].get('active_team') == 2:
                        if matches[i].get('player_slot') % 2 == 0:
                            rad = 0
                        else:
                            rad = 1
                    else:
                        if matches[i].get('player_slot') % 2 == 0:
                            rad = 1
                        else:
                            rad = 0
                    choose = False

                if matches[i].get('player_slot') % 2 == rad:
                    str_radiant += str(matches[i].get('hero_id'))
                    j += 1
                    if j < 5:
                        str_radiant += ", "
                else:
                    str_dire += str(matches[i].get('hero_id'))
                    k += 1
                    if k < 5:
                        str_dire += ", "

        # print (match_id[num])
        # print (num)
        # print([str_radiant, str_dire])
        match_heroes.append(np.asarray([str_radiant, str_dire], dtype=list))
    match_heroes = np.asarray(match_heroes, dtype=list)
    return match_heroes