class Node (object):
	"""docstring for Tree """
	def __init__(self, label=None):
		# super(Tree , self).__init__()
		self.label = label
		self.left = None
		self.right = None

# string = ''
def printTree(T):
	def recur(T):
		if T.label!=None :
			return str(T.label)+' '
		else :
			string = ''
			string+=recur(T.left)
			string+=recur(T.right)
			return string
	return recur(T)

def getITGPermut(lb,ub):
	'''
	params lb-lower bound
	params ub-upper bound
	return set of itg-constrained string
	'''
	def recur(start, end):
		if end - start <=1:
			return [Node(start)]
		else :
			all_t = []
			for k in xrange(start+1,end):
				for i in recur(start,k):
					for j in recur(k,end):
						for l,r in [(i,j),(j,i)]:
							t = Node()
							t.left = l
							t.right= r
							all_t.append(t)
			return all_t

	allString = set()
	allItg = recur(lb,ub)
	for t in allItg:
		allString.add(printTree(t))
	return allString

def permut_phrases(phrase_pairs, itg_permut):
	'''
	phrase_pairs is list of phrase_pair to be itg-constructed
	itg-permut is list of list of integer permutation
	we will permute the english side of pairs, assume right side of tuple
	return : list of tuple contain permuted join phrases
	'''
	all_permuted = []
	for itg in itg_permut:
		#allocate list
		right = ['' for i in range(len(phrase_pairs))]
		left = []
		itglist = itg.split()
		for i in range(len(itglist)):
			left.append(phrase_pairs[i][0])
			right[itglist[i]] = phrase_pairs[i][1]
		all_permuted.append((' '.join(left),' '.joint(right)))
	return all_permuted

def devide_phrases_accto_length(all_phrase_pairs):
	#dictionary with length as key and list of phrase_pairs as value
	for p in all_phrase_pairs:
		








if __name__ == '__main__':
	allString = getITGPermut(1,5)
	print len(allString)
	for s in allString:
		print s


		