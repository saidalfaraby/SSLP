from collections import defaultdict
from extract_phrases import load_phrases
from itertools import permutations, product
import dill as pickle
import sys
import time



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
	log('construct ITG permutation of length '+str(ub))
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
			right[int(itglist[i])] = phrase_pairs[i][1]
		all_permuted.append((' '.join(left), ' '.join(right)))
	return set(all_permuted)


def devide_phrases_accto_length(all_phrase_pairs, joint_phrase_table, threshold=100):
	'''
	return dictionary with length as key and list of phrase_pairs as value
	we will take max length between english and foreign phrase
	'''
	log('devide phrase pairs according to length ...')

	div_phrases = defaultdict(list)

	if threshold=='average':
		max_prob_per_length = defaultdict(lambda : 0)
		sum_prob_per_length = defaultdict(list)
		for pp in joint_phrase_table:
			l = max(len(pp[0].split()), len(pp[1].split()))
			sum_prob_per_length[l].append(joint_phrase_table[pp])
		for k in sum_prob_per_length:
			max_prob_per_length[k] = sum(sum_prob_per_length[k])/len(sum_prob_per_length[k])
		log('max prob per length'+str(max_prob_per_length))
		for p in all_phrase_pairs:
			pl = max(len(p[0].split()), len(p[1].split()))
			if joint_phrase_table[p]>= max_prob_per_length[pl]:
				div_phrases[pl].append(p)
	elif isinstance(threshold,int):
		#take top threshold most probable phrases for each length
		topThreshold = defaultdict(list)
		for pp in joint_phrase_table:
			l = max(len(pp[0].split()), len(pp[1].split()))
			topThreshold[l].append((pp,joint_phrase_table[pp]))
			topThreshold[l]= sorted(topThreshold[l],key=lambda x: x[1])
			topThreshold[l] = topThreshold[l][:threshold]
		for l in topThreshold:
			for p in topThreshold[l]:
				div_phrases[l].append(p[0])

	le = [len(div_phrases[l]) for l in div_phrases]
	log('for all length '+str(le))
	log('sum of all length '+str(sum(le)))
	log('original phrases length '+str(len(all_phrase_pairs)))
	f = open('chosen_phrases.txt', 'a+')
	for l in div_phrases:
		f.write("Phrases of length : "+str(l)+'\n')
		f.write("-------------------------\n")
		for i in div_phrases[l]:
			f.write(str(i)+'\n')
		f.write('\n\n')
	f.close()
	return div_phrases


def construct_itg_phrases(all_phrase_pairs, joint_phrase_table, threshold=100, max_length = 4):
	st = time.time()
	log('construct ITG phrases with threshold='+str(threshold)+' and max_length='+str(max_length))
	div_phrases = devide_phrases_accto_length(all_phrase_pairs, joint_phrase_table, threshold)
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
	log('start combining phrases with ITG-permutation ...')
	for l in length_dict:
		itgPermut = getITGPermut(0,l)
		for c in length_dict[l]:
			log('processing combination of : '+str(c))
			setC = set()
			temp =[range(len(div_phrases[u])) for u in c]
			indices_of_u = product(*temp)
			for i in indices_of_u:
				list_phrases = []
				for ii in range(len(i)):
					list_phrases.append(div_phrases[c[ii]][i[ii]])
				setC.update(permut_phrases(list_phrases, itgPermut))
			log('number of combined phrases : '+str(len(setC)))
			all_phrase_pairs.update(setC)
	log('New phrases size : '+str(len(all_phrase_pairs)))
	log('Time to construct itg phrases : '+str(time.time()-st))
	return all_phrase_pairs


def save_phrases(phrase_pairs, folder=''):
    log('Saving...')
    with open(folder+'combined_phrase_pairs_.pickle', 'wb') as handle:
        pickle.dump(phrase_pairs, handle)

def log(string):
	f = open('log.txt', 'a+')
	print string
	f.write(string+'\n')
	f.close()

def getAllLengthComb(maxLength):
	log('construct all possible length combinations of length '+str(maxLength))
# def all_length_comb(Length):
	def findAllSumTo(N):
		final = []
		ind = range(1,N+1)
		for j in ind:
			choose = Array(j)
			Next = N-j
			if Next > 0:
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
	log('return all length combinations ...')
	return allComb


if __name__ == '__main__':
	folder  = 'training/'
	if len(sys.argv) > 1:
		folder = sys.argv[1]
	log('load file from folder : '+folder)
	phrase_pairs, en_given_nl, nl_given_en, joint_ennl = load_phrases(folder)
	combined_phrase_pairs = construct_itg_phrases(phrase_pairs, joint_ennl, threshold=5, max_length=3)
	print combined_phrase_pairs
	save_phrases(combined_phrase_pairs, folder)
	log('--------------------------------\n\n')
	# print getAllLengthComb(4)
