#EE2703
#Assingment 2
# The task is given in the pdf.
import sys
import numpy as np
no_arg=len(sys.argv) #number of arguments received in the command line
# The argument is the filename of the netlist file
arg1=sys.argv[1]
with open(arg1) as f:
	lines=f.readlines() #Gets all the lines in the document in a list, lines
i=0
flag=0
omega=0
e_cnt=0 #number of edges
adj=[]

#An object of this class is an edge between two nodes connected by any circuit element.
class edge: 
	def __init__(self,type,othernode,value,sgn):
		self.other_node=othernode
		self.type=type
		self.id=e_cnt
		self.sign=sgn
		self.vis=False #visited or not.
		if type=='R' or type=='V' or type=='I':
			self.value=complex(value)
		elif type=='L':
			self.value=complex(value)*omega*(1j)
		elif type=='C':
			if omega!=0:
				self.value=1/(complex(value)*omega*(1j));
			else:
				self.value=np.inf


node_inp_pro={} #Maps the input node name with the node number as stored by the program
node_pro_inp={} #Maps the node number used by the program with the node name
node_inp_pro['GND']=0
node_pro_inp[0]='GND'

for s in lines: #s is each line of the netlist file
	i+=1
	if s[0:8] ==".circuit": #When all the initial comments are bypassed; actual circuit definition begins
		flag=1
		break
if flag==1:
	a="abc";
	data=[]
	while a[0:4]!=".end": #No '.end' found in the file
		if(i>=len(lines)):
			flag=0
			break
		l_ins=[]
		a=lines[i]
		i+=1
		arr=a.split()
		reqlen=0
		if arr[0][0]=='R' or arr[0][0]=='L' or arr[0][0]=='C': 
			reqlen=4
		elif  arr[0][0]=='V':
			if arr[3]=='ac': #If the input is an ac source, we have to use complex numbers for analysis
				arr[3]=complex(arr[4])+(complex(arr[5])*(1j))
				# print("hello")
			else:
				arr[3]=arr[4]
				# print("hello2")
			reqlen=4
		elif  arr[0][0]=='I':
			if arr[3]=='ac': #If the input is an ac source, we have to use complex numbers for analysis
				arr[3]=complex(arr[4])+(complex(arr[5])*(1j))
			else:
				arr[3]=arr[4]
			reqlen=4
		elif arr[0][0]=='E' or arr[0][0]=='G':
			reqlen=6
		elif arr[0][0]=='H' or arr[0][0]=='F':
			reqlen=5
		else:
			continue
		for n in range (0,reqlen,1):
			l_ins.append(arr[n])
		data.append(l_ins)
	while a[0:3]!=".ac" and i<len(lines):
		a=lines[i]
		i+=1
		# print("Hi",a)
		if a[0:3]==".ac":
			arr=a.split()
			omega=complex(arr[2])*2*np.pi
			break
	# if flag!=0: #Idk why I put this line in assign1.
	# 	for s in data:
	# 		for t in s:
	# 			print(t,end=" ")
	# 		print("\n",end="")
	# print(data)
	# print(omega)
	idx=1
	for arr in data: #Add data to the maps
		#if arr[0][0]=='R' or arr[0][0]=='L' or arr[0][0]=='C' or arr[0][0]=='V' or arr[0][0]=='I' or arr[0][0]=='H' or arr[0][0]=='F' or arr[0][0]=='E' or arr[0][0]=='G': 
		if arr[1] not in node_inp_pro:
			node_inp_pro[arr[1]]=idx
			node_pro_inp[idx]=arr[1]
			idx+=1
		if arr[2] not in node_inp_pro:
			node_inp_pro[arr[2]]=idx
			node_pro_inp[idx]=arr[2]
			idx+=1
		if arr[0][0]=='E' or arr[0][0]=='G':
			if arr[3] not in node_inp_pro:
				node_inp_pro[arr[3]]=idx
				node_pro_inp[idx]=arr[3]
				idx+=1
			if arr[4] not in node_inp_pro:
				node_inp_pro[arr[4]]=idx
				node_pro_inp[idx]=arr[4]
				idx+=1
	n_nodes=len(node_pro_inp)
	n_edges=len(data)
	for i in range (n_nodes):
		adj.append([])
	#Creating adjacency list:
	for arr in data: #Assuming no controlled sources
		e=edge(arr[0][0],node_inp_pro[arr[2]],arr[3],1)
		adj[node_inp_pro[arr[1]]].append(e)
		f=edge(arr[0][0],node_inp_pro[arr[1]],arr[3],(-1))
		adj[node_inp_pro[arr[2]]].append(f)
		e_cnt+=1
	# print(node_inp_pro)
	#Laws (or making equations by KVL, KCL): 
	B=np.full((0,1),0.0+0.0j)
	A=np.full((0,n_nodes+n_edges),0.0+0.0j) #one row of zeroes. Concatenate the subsequent rows to this.
	# B=np.full((1,1),0.0+0.0j)
	# A=np.full((1,n_nodes+n_edges),0.0+0.0j) #one row of zeroes. Concatenate the subsequent rows to this.
	# A[0,0]=1.0
	completed_nodes=[]
	for i in range(n_nodes):	
		if i in completed_nodes:
			continue
		# print("Node number:",i)	 
		n_eqn=np.full((1,n_nodes+n_edges),0.0+0.0j)
		b_n=np.full((1,1),0.0+0.0j)
		supernodes=[]
		supernodes.append(i)
		completed_nodes.append(i)
		while len(supernodes)>0:
			nod=supernodes.pop(0)
			completed_nodes.append(nod)
			for e in adj[nod]: #iterating through the adjacency list of node i.
				if e.vis==False: #visited is false means that this edge is not processed before
					# print(e.type,e.value,e.id,e.other_node,e.sign)
					e_eqn=np.full((1,n_nodes+n_edges),0.0+0.0j)
					b_e=np.full((1,1),0.0+0.0j)
					if e.type=='L' or e.type=='C' or e.type=='R':
						e.vis=True
						e_eqn[0,e.other_node]=1/(e.value)
						e_eqn[0,nod]=-1/(e.value)
						e_eqn[0,n_nodes+e.id]=(-1)*e.sign
						n_eqn[0,e.other_node]+=(1/e.value)
						n_eqn[0,nod]+=(-1/e.value)
					elif e.type=='V':
						e.vis=True
						e_eqn[0,e.other_node]=1
						e_eqn[0,nod]=-1
						b_e[0,0]=e.value*e.sign
						if e.other_node>nod: #To ensure that a pair of nodes are processed only once.
							supernodes.append(e.other_node)
							e_eqn1=np.full((1,n_nodes+n_edges),0.0+0.0j) #To deal with the current in that branch.
							b_e1=np.full((1,1),0.0+0.0j)
							for ef in adj[nod]:
								e_eqn1[0,n_nodes+ef.id]=1*ef.sign
							# b_e1=np.full((1,1),1.0+0.0j)
							# e_eqn1[0,n_nodes+e.id]=1
							A=np.concatenate((A,e_eqn1),axis=0)
							B=np.concatenate((B,b_e1),axis=0)
					elif e.type=='I':
						e.vis=True
						e_eqn[0,n_nodes+e.id]=1
						b_e[0,0]=e.value*e.sign
						b_n+=(e.sign*e.value)
					if e.other_node>nod:
						# print(e_eqn)
						# print(b_e)
						A=np.concatenate((A,e_eqn),axis=0)
						B=np.concatenate((B,b_e),axis=0)
		# print(n_eqn)
		# print(b_n)
		A=np.concatenate((A,n_eqn),axis=0)
		B=np.concatenate((B,b_n),axis=0)
	A=np.delete(A,-1,0)
	B=np.delete(B,-1,0)
	A_eqn=np.full((1,n_nodes+n_edges),0.0+0.0j)
	b_eqn=np.full((1,1),0.0+0.0j)
	A=np.concatenate((A,A_eqn),axis=0)
	B=np.concatenate((B,b_eqn),axis=0)
	A[-1,0]=1.0
	#Now we have the matrices ready. Ax=b, x is the vector of all branch currents and the node voltages
	# print(A,b)
	# print(data[1])
	# print(node_inp_pro)
	# print(node_pro_inp)
	# print(A)
	# print(B)
	# print(A.shape)
	# print(B.shape)
	# print(n_nodes,n_edges)
	sol_n=np.linalg.solve(A,B) 
	# print(sol_n)
	# print(adj)
	#sol_n has the required output as specified in the problem statement.
	for i in range(n_nodes):
		print("Voltage at node",i,"(",node_pro_inp[i],") =",sol_n[i][0])
	print("The currents are:")
	for i in range(n_edges):
		print("I_",i,"=",sol_n[i+n_nodes][0])
if flag==0:
	print("Malformed input file!")
