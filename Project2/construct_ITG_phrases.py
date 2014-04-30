from collections import defaultdict
from extract_phrases import load_phrases


class Node (object):
    """docstring for Tree """
    def __init__(self, label=None):
		# super(Tree , self).__init__()
		self.label = label
		self.left = None
		self.right = None

class Array(object):
	"""docstring for Array"""
	def __init__(self, label=None):
		self.label = label
		self.children = []
		self.parent = None
		


def printTree(T):
	# def recur(T):
	if T.label is not None:
		return str(T.label)+' '
	else:
		string = ''
		string += printTree(T.left)
		string += printTree(T.right)
		return string
	# return recur(T)


def getITGPermut(lb, ub):
	'''
	lb : lower bound of permuting integer
	ub : upper bound of permuting integer
	return set of itg-constrained string
	'''
	def recur(start, end):
		if end - start <= 1:
			return [Node(start)]
		else:
			all_t = []
			for k in xrange(start+1, end):
				for i in recur(start, k):
					for j in recur(k, end):
						for l, r in [(i, j), (j, i)]:
							t = Node()
							t.left = l
							t.right = r
							all_t.append(t)
			return all_t

	allString = set()
	allItg = recur(lb, ub)
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
		all_permuted.append((' '.join(left), ' '.joint(right)))
	return all_permuted


def devide_phrases_accto_length(all_phrase_pairs, joint_phrase_table):
	'''
	return dictionary with length as key and list of phrase_pairs as value
	we will take max length between english and foreign phrase
	'''
	div_phrases = defaultdict(list)

	max_prob_per_length = defaultdict(lambda : 0)
	sum_prob_per_length = defaultdict(list)
	for pp in joint_phrase_table:
		l = max(len(pp[0].split()), len(pp[1].split()))
		# if joint_phrase_table[pp]>= max_prob_per_length[l]:
			# max_prob_per_length[l] = joint_phrase_table[pp]
		sum_prob_per_length[l].append(joint_phrase_table[pp])
	for k in sum_prob_per_length:
		max_prob_per_length[k] = sum(sum_prob_per_length[k])/len(sum_prob_per_length[k])


	print 'max prob per length', max_prob_per_length


	for p in all_phrase_pairs:
		pl = max(len(p[0].split()), len(p[1].split()))
		if joint_phrase_table[p]>= max_prob_per_length[pl]:
			div_phrases[pl].append(p)
	le = [len(div_phrases[l]) for l in div_phrases]
	print 'for all length ',le
	print 'sum of all length ', sum(le)
	print 'original length ', len(all_phrase_pairs)
	return div_phrases


def construct_itg_phrases(all_phrase_pairs):
	div_phrases = devide_phrases_accto_length(all_phrase_pairs)
	max_length = 4
	#find all combination of length that sum to 2 up to max_length
	length_combinations = []
	for l in range(2, max_length+1):
		length_combinations += getAllLengthComb(l)
	#devide length_combination into number of part ex : from 2 parts up to 4 parts
	#save into dictionary with number of parts as the key
	length_dict = defaultdict(list)
	for lc in length_combinations:
		length_dict[len(lc)].append(lc)

	#there is a more efficient way by using smaller part, but going for correctness first
	for l in length_dict.keys():
		itgPermut = getITGPermut(0,l)
		for c in length_dict[l]:
			pass


def getAllLengthComb(maxLength):
# def all_length_comb(Length):
	def findAllSumTo(N):
		final = []
		ind = range(1,N+1)
		for j in ind:
			choose = Array(j)
			Next = N-j
			if N > 0:
				ss =findAllSumTo(Next)
				for i in ss: 
					choose.children.append(i)
					i.parent = choose
			final.append(choose)
		return final
		# return findAllSumTo(Length)

	tree = findAllSumTo(maxLength)

	def getAllPath(T):
		allLeaves = []
		#recur All leaves
		def recur(T):
			if len(T.children)==0:
				allLeaves.append(T)
			else :
				for i in (T.children):
					recur(i)
		recur(T)

		#for each leaves, travel to root until parent = None
		allcomb = []
		for leave in allLeaves:
			l = leave
			comb = [l.label]
			while l.parent!=None:
				l = l.parent
				comb.append(l.label)
			if len(comb)>1:
				allcomb.append(comb)
		return allcomb

	allComb = []
	for t in tree :
		allComb += getAllPath(t)
	return allComb


if __name__ == '__main__':
	# allString = getITGPermut(1,5)
	# print len(allString)
	# for s in allString:
	# 	print s
	# t = findAllSumTo(4)
	# print len(t)
	# test = []
	# for i in t:
	# 	test+=printTest(i)
	# print test

	phrase_pairs, en_given_nl, nl_given_en, joint_ennl = load_phrases('training/')

	devide_phrases_accto_length(phrase_pairs, joint_ennl)

	t =getAllLengthComb(4)
	print t

	t = getITGPermut(0,4)
	print t
