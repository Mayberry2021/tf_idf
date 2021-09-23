from DTM import *

section_list = ['politics','economy','society','life','world','it']

db_list = [file_name for file_name in os.listdir(os.getcwd()) if file_name.endswith('.db')]
print(db_list)

lab_data = []
for section in section_list:
	temp = train_set(db_list[0], section)
	lab_data += temp

titles = text_slice(lab_data)

input_data = []
for section in section_list:
	temp = train_set(db_list[1], section)
	input_data += temp

input_title = text_slice(input_data)

for title in input_title:
	most_similar(titles, title)