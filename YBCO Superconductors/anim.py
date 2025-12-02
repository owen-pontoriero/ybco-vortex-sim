import time
import sys
import os

os.system('clear')

#~~~~~~~~~~~~~~~~~~~~~|Loading Screen|~~~~~~~~~~~~~~~~~~~~~~~~~

print("\n")
os.system('python LoadingScreen.py')

#~~~~~~~~~~~~~~~~~~~~~|File Verifier|~~~~~~~~~~~~~~~~~~~~~~~~~

print("\nWould you like to check your data files?")

while True:
	choice = input("Input Y/N → ").strip().lower()

	if choice == 'y':
		os.system('clear')
		print('Checking Files')
		time.sleep(1.5)
		print ("Running CheckCSV.py...")
		#run file
		os.system('python LoadingScreen.py')
		os.system('clear')
		os.system("python anim2.py")
		break

	if choice == 'n':
		os.system('clear')
		print ('Okay, will not check files')
		time.sleep(1.5)
		break
	else:
		print("Invalid answer, please try again.")

os.system('clear')

#~~~~~~~~~~~~~~~~~~~~|What Simulation?|~~~~~~~~~~~~~~~~~~~~~~~~~

print("Which simulation would you like to run?")

while True:

	choice1 = input("\n2D or 3D? (Type 2 or 3) \nType Exit to Exit\n\nInput Your Answer → ").strip().lower()
	if choice1 == "exit":
		break
		os.system('exit')
	os.system('clear')
	
	print("Which simulation would you like to run?")
	
	choice2 = input("\n2D: 1/2/3/4\n3D: 1/2 \n\nType Exit to Exit\n\nInput Your Answer → ").strip().lower()
	if choice2 == "exit":
		break
		os.system('exit')

	if choice1 == '2':
		os.system('clear')
		print('Running 2D Simulation')
		time.sleep(1.5)
		
		#run file
		os.system('clear')
		#os.system('python LoadingScreen.py')
		os.system(f'cd ~/downloads/YBCO\ Superconductors/ReadCSV && python ReadCSV_{choice2}.py')

	if choice1 == '3':
		os.system('clear')
		print('Running 3D Simulation')
		time.sleep(1.5)
		
		#run file
		os.system('clear')
		#os.system('python LoadingScreen.py')
		os.system(f'cd ~/downloads/YBCO\ Superconductors/ReadCSV && python Anim3D_{choice2}.py')

	else:
		print("Invalid answer, please try again.")


#~~~~~~~~~~~~~~~~~~~~~~~~~|End|~~~~~~~~~~~~~~~~~~~~~~~~~~~

