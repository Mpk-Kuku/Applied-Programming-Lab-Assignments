#Assign 1
# The task is the same as the one given in the pdf.
import sys
no_arg=len(sys.argv) #number of arguments received in the command line
# The argument is the filename of the netlist file
arg1=sys.argv[1]
with open(arg1) as f:
	lines=f.readlines() #Gets all the lines in the document in a list, lines
i=0
flag=0
for s in lines: #s is each line of the netlist file
	i+=1
	if s[0:8] ==".circuit": #When all the initial comments are bypassed; actual circuit definition begins
		flag=1
		break
if flag==1:
	a="abc";
	l1=[]
	while a[0:4]!=".end":
		if(i>=len(lines)):
			flag=0 #No '.end' found in the file
			break
		l_ins=[]
		a=lines[i]
		i+=1
		arr=a.split() #Splits the spaces
		reqlen=0
		if arr[0][0]=='R' or arr[0][0]=='L' or arr[0][0]=='C' or arr[0][0]=='V' or arr[0][0]=='I': 
			reqlen=4 #The first 4 are the values that matter in the circuit.
		elif arr[0][0]=='E' or arr[0][0]=='G':
			reqlen=6 #The first 6 are the values that matter in the circuit.
		elif arr[0][0]=='H' or arr[0][0]=='F':
			reqlen=5 #The first 5 are the values that matter in the circuit.
		else:
			continue
		for n in range (reqlen-1,-1,-1):
			l_ins.append(arr[n]) #Added into a list in reverse order
		l1.append(l_ins) #The list is added to another list
	l1.reverse() #The list of lists is reversed to get the required order to output.
	if flag!=0: #Printing the output:
		for s in l1: 
			for t in s:
				print(t,end=" ")
			print("\n",end="")
if flag==0: #No '.circuit' or '.end' in the netlist file
	print("Malformed input file!")
