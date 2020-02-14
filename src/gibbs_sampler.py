# -*- coding: utf-8 -*-

"""
    File name: gibbs_sampler.py
    Description: a re-implementation of the Gibbs sampler for http://www.gatsby.ucl.ac.uk/teaching/courses/ml1
    Date created: December 2019
    Python version: 3.6
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)



# todo: sample everything from self.rang_gen to control the random seed (works as numpy.random)
class GibbsSampler:
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          number of documents
        :param n_topics:        number of topics
        :param n_words:         number of words in vocabulary
        :param alpha:           dirichlet parameter on topic mixing proportions
        :param beta:            dirichlet parameter on topic word distributions
        :param random_seed:     random seed of the sampler
        """
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.n_words = n_words
        self.alpha = alpha
        self.beta = beta
        self.rand_gen = np.random.RandomState(random_seed)

        self.docs_words = np.zeros((self.n_docs, self.n_words))
        self.docs_words_test = None
        self.loglike = None
        self.loglike_test = None
        self.do_test = False

        self.A_dk = np.zeros((self.n_docs, self.n_topics))  # number of words in document d assigned to topic k
        self.B_kw = np.zeros((self.n_topics, self.n_words))  # number of occurrences of word w assigned to topic k
        self.A_dk_test = np.zeros((self.n_docs, self.n_topics))
        self.B_kw_test = np.zeros((self.n_topics, self.n_words))

        self.theta = np.ones((self.n_docs, self.n_topics)
                             ) / self.n_topics  # theta[d] is the distribution over topics in document d, nomalised so that row sum =1
        self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words  # phi[k] is the distribution words in topic k, nomalised so that row sun=1

        self.topics_space = np.arange(self.n_topics)
        self.topic_doc_words_distr = np.zeros((self.n_topics, self.n_docs, self.n_words))  # z_id|x_id, theta, phi

    def init_sampling(self, docs_words, docs_words_test=None,
                      theta=None, phi=None, n_iter=0, save_loglike=False):
        assert np.all(docs_words.shape == (self.n_docs, self.n_words)), "docs_words shape=%s must be (%d, %d)" % (
            docs_words.shape, self.n_docs, self.n_words)
        self.n_docs = docs_words.shape[0]

        self.docs_words = docs_words
        self.docs_words_test = docs_words_test

        self.do_test = (docs_words_test is not None)

        if save_loglike:
            self.loglike = np.zeros(n_iter+1)             #n_iter is the number of iteration, plus one initial loglikelihood

            if self.do_test:
                self.loglike_test = np.zeros(n_iter+1)


        self.A_dk.fill(0.0)
        self.B_kw.fill(0.0)
        self.A_dk_test.fill(0.0)
        self.B_kw_test.fill(0.0)

        self.init_params(theta, phi)

    def init_params(self, theta=None, phi=None):
        if theta is None:
            self.theta = np.ones((self.n_docs, self.n_topics)) / self.n_topics
        else:
            self.theta = theta.copy()

        if phi is None:
            self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words
        else:
            self.phi = phi.copy()

        self.update_topic_doc_words()
            #print('z_id',self.topic_doc_words_distr)
        self.sample_counts()

    def run(self, docs_words, docs_words_test=None,
            n_iter=100, theta=None, phi=None, save_loglike=False):
        """
        docs_words is a matrix n_docs * n_words; each entry
        is a number of occurrences of a word in a document
        docs_words_test does not influence the updates and is used
        for validation
        """
        self.init_sampling(docs_words, docs_words_test,
                           theta, phi, n_iter, save_loglike)
        self.update_loglike(-1)                             #record the initial loglikelihood
        self.update_loglike_test(docs_words_test,-1)        #record the initial testing loglikelihood
        for iteration in range(n_iter):
            self.update_params()
                                    #print('z_id', self.topic_doc_words_distr)
                                    #print('A_dk', self.A_dk)
            if save_loglike:
                self.update_loglike(iteration)
                self.update_loglike_test(docs_words_test, iteration)   #testing loglikelihood
                                    #print("likelihood ", self.loglike)
        return self.to_return_from_run()

    def to_return_from_run(self):
        return self.topic_doc_words_distr, self.theta, self.phi

    def update_params(self):
        """
        Samples theta and phi, then computes the distribution of
        z_id and samples counts A_dk, B_kw from it
        """
        # todo: sample theta and phi

        #sample theta from dirichlet (A_{d,k}+alpha), since dim(theta)=ndoc * ntopic , we need to update for each d, so for each row
        for d in range(self.n_docs):
            self.theta[d,:] = np.random.dirichlet(self.A_dk[d,:] + self.alpha)

        #sample phi from dirichlet (B_{k,w}+beta), dim(phi) = ntopics * nwords
        for k in range(self.n_topics):
            self.phi[k,:] = np.random.dirichlet(self.B_kw[k,:] + self.beta)


        self.update_topic_doc_words()
        #print('thishif',self.topic_doc_words_distr[0,0,:])
        self.sample_counts()        #update A and B

    def update_topic_doc_words(self):
        """
        Computes the distribution of z_id|x_id, theta, phi
        """
        self.topic_doc_words_distr = np.repeat(
            self.theta.T[:, :, None], self.n_words, axis=2) * self.phi[:, None, :]
        self.topic_doc_words_distr /= self.theta.dot(self.phi)[None, :, :]

    def sample_counts(self):
        """
        For each document and each word, samples from z_id|x_id, theta, phi
        and adds the results to the counts A_dk and B_kw
        """

        self.A_dk.fill(0)
        self.B_kw.fill(0)

        if self.do_test:
            self.A_dk_test.fill(0)
            self.B_kw_test.fill(0)

        # todo: sample a topic for each (doc, word) and update A_dk, B_kw correspondingly
        self.docs_topic  = np.zeros((self.n_docs,self.n_words))       #np.zeros((self.n_docs, self.n_words))
        for docs in range(self.n_docs):
            for word in range(self.n_words):
                for _ in range(self.docs_words[docs, word]):           #for number of occurance of the word
                    sample_topic = np.random.choice(self.topics_space,p = self.topic_doc_words_distr[:,docs,word])  #sample topic for each word in each document
                    self.A_dk[docs, sample_topic] += 1
                    self.B_kw[sample_topic, word] += 1
            #update A_dk
            #print(self.docs_topic[docs,:])
            #unique , counts = np.unique(self.docs_topic[docs,:],return_counts = True)
            #for topic in range(self.n_topics):
             #   self.A_dk[docs,topic] = np.count_nonzero(self.docs_topic[docs,:] == topic)

        #update  B_kw   np.zeros((self.n_topics, self.n_words))

        #for topic in range(self.n_topics):
         #   for word in range(self.n_words):
          #      self.B_kw[topic , word] = np.count_nonzero(self.docs_topic[:,word] == topic )

        pass

    def update_loglike(self, iteration):
        """
        Updates loglike of the data, omitting the constant additive term
        with Gamma functions of hyperparameters
        """
        # todo: implement log-like
        # Hint: use scipy.special.gammaln (imported as gammaln) for log(gamma)

        #theta prior and phi prior
        theta_prior = self.n_docs * (gammaln(self.n_topics * self.alpha) - self.n_docs * self.n_topics * gammaln(self.alpha))  #gammaln is the log of gamma function
        phi_prior = self.n_topics * (gammaln(self.n_words * self.beta) - self.n_topics * self.n_words * gammaln(self.beta))

        # p(z_id) and p(x_id)
        p_zid = np.sum(np.multiply((self.A_dk + self.alpha -1) , np.log(self.theta)))
                #print('p_zid',self.p_zid)
        p_xid = np.sum(np.multiply((self.B_kw + self.beta -1) , np.log(self.phi)))
                #   print("p_xid",self.p_xid)
        self.loglike[iteration+1] = theta_prior + phi_prior + p_zid + p_xid     #plus one since the first one is the initial loglike
        #print("loglikelihood is ", self.loglike)
        '''
        ll = 0
        ll += self.n_topics * gammaln(self.n_words * self.beta)
        ll += - self.n_words * self.n_topics * gammaln(self.beta)
        ll += self.n_docs * gammaln(self.n_topics * self.alpha)
        ll += - self.n_topics * self.n_docs * gammaln(self.alpha)
        ll += ((self.A_dk + self.alpha -1 ) * np.log(self.theta)).sum()
        ll += ((self.B_kw + self.beta - 1) * np.log(self.phi)).sum()
        self.loglike[iteration + 1] = ll
        '''

        pass

    def update_loglike_test(self, docs_words_test, iteration):
        """
        Update testing loglikelihood, using the testing data, theta and phi
        """
        for docs in range(self.n_docs):
            for word in range(self.n_words):
                p = np.matmul(self.theta[docs,:], self.phi[:,word])
                if p!=0:
                    self.loglike_test[iteration+1] += docs_words_test[docs, word] * np.log(p)
        pass

    def get_loglike(self):
        """Returns log-likelihood at each iteration."""
        #self.ll = np.append(self.ll, self.loglike)
        if self.do_test:
            return self.loglike, self.loglike_test
        else:
            return self.loglike


class GibbsSamplerCollapsed(GibbsSampler):
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          number of documents
        :param n_topics:        number of topics
        :param n_words:         number of words in vocabulary
        :param alpha:           dirichlet parameter on topic mixing proportions
        :param beta:            dirichlet parameter on topic word distributions
        :param random_seed:     random seed of the sampler
        """
        super().__init__(n_docs, n_topics, n_words, alpha, beta, random_seed)   # __init__ in GibbsSampler

        # topics assigned to each (doc, word)
        self.doc_word_samples = np.ndarray((self.n_docs, self.n_words), dtype=object)
        self.doc_word_samples_test = self.doc_word_samples.copy()

    def init_params(self, theta=None, phi=None):
        # z_id are initialized uniformly
        for doc in range(self.n_docs):
            for word in range(self.n_words):
                if self.do_test:
                    additional_samples = self.docs_words_test[doc, word]
                else:
                    additional_samples = 0

                sampled_topics = self.rand_gen.choice(self.topics_space, size=self.docs_words[doc, word] + additional_samples)

                sampled_topics_train = sampled_topics[:self.docs_words[doc, word]]      #.docs_words is the how many times the word occurs
                self.doc_word_samples[doc, word] = sampled_topics_train.copy()          # now each cell is an np.array !!!,

                sample, counts = np.unique(sampled_topics_train, return_counts=True)

                self.A_dk[doc, sample] += counts        #for each element
                self.B_kw[sample, word] += counts

                if self.do_test:
                    sampled_topics_test = sampled_topics[self.docs_words[doc, word]:]
                    self.doc_word_samples_test[doc, word] = sampled_topics_test.copy()

                    sample, counts = np.unique(sampled_topics_test, return_counts=True)

                    self.A_dk_test[doc, sample] += counts
                    self.B_kw_test[sample, word] += counts


    def update_params(self):
        """
        Computes the distribution of z_id.
        Sampling of A_dk, B_kw is done automatically as
        each new z_id updates these counters
        """
        # todo: sample a topic for each (doc, word) and update A_dk, B_kw correspondingly
        # Hint: you can update A_dk, B_kw after each sampling instead of re-computing the whole matrix

        #self.docs_topic_col =  np.ndarray(self.n_docs, self.n_words)
        for docs in range(self.n_docs):
            for word in range(self.n_words):
                for index_of_oldtopic ,old_topic in enumerate(self.doc_word_samples[docs,word]):          #self.docs_word_samples is z_id


                    #print("\n old A=", self.A_dk)
                    #print("the old topic=", old_topic)

                    self.A_dk[docs, old_topic] -= 1 #if self.A_dk[docs, old_topic] >= 1 else 0)
                    self.B_kw[old_topic, word] -= 1 #if self.B_kw[old_topic, word] >= 1 else 0)

                    #print("\n A after minus one", self.A_dk)

                    p = (self.A_dk[docs, :] + self.alpha) * ((self.B_kw[:,word]+self.beta)/np.sum(self.B_kw, axis=1))   #might have mistake

                    p /= np.sum(p)    #the probability of topic of i th word in j th document

                    #print("\n the p is ", p)

                    new_topic = np.random.choice(len(p), p= p)       #the sampled new topic for thid word

                    #print("\n the new topic is", new_topic)

                    self.doc_word_samples[docs,word][index_of_oldtopic] = new_topic    #update the new topic
                    self.A_dk[docs, new_topic] += 1
                    self.B_kw[new_topic, word] += 1             #update A and B

                    #print("\n the updated A is", self.A_dk)

        pass

    def update_loglike(self, iteration):
        """
        Updates loglike of the data, omitting the constant additive term
        with Gamma functions of hyperparameters
        """
        # todo: implement log-like, this is different from the standard LDA
        ll = 0
        ll += self.n_docs * gammaln(self.n_topics * self.alpha)
        ll += - self.n_topics * self.n_docs * gammaln(self.alpha)
        ll += self.n_topics * gammaln(self.n_words * self.beta)
        ll += - self.n_words * self.n_topics * gammaln(self.beta)
        ll += gammaln(self.A_dk + self.alpha).sum()
        ll += - gammaln(np.sum(self.A_dk,axis = 1) + self.n_topics * self.alpha).sum()
        ll += gammaln(self.B_kw + self.beta).sum()
        ll += - gammaln(np.sum(self.B_kw, axis = 1) + self.n_words * self.beta).sum()


        self.loglike[iteration+1] = ll    #plus one since the first one is the initial loglike

        pass

    def update_loglike_test(self, docs_words_test, iteration):
        """
        Updating the testing log likelihood using alpha, A, B, beta, and formula (20) in 1(d)

        """
        for docs in range(self.n_docs):
            for word in range(self.n_words):
                p1 = self.alpha + self.A_dk[docs, :] -1
                p2 = self.n_topics * self.alpha -1 + self.A_dk[docs, :].sum()
                p3 = self.B_kw[:,word] + self.beta - 1
                p4 = self.B_kw.sum(axis = 1) + self.n_words * self.beta -1
                p11 = p1/p2
                p22 = p3/p4
                p = np.matmul(p11, p22)
                #print("p=",p)
                if p!=0:
                    self.loglike_test[iteration+1] += docs_words_test[docs, word] * np.log(p)



    def to_return_from_run(self):
        return self.doc_word_samples



def read_data(filename):
    """
    Reads the text data and splits into train/test.
    Examples:
    docs_words_train, docs_words_test = read_data('./code/toyexample.data')
    nips_train, nips_test = read_data('./code/nips.data')
    :param filename:    path to the file
    :return:
    docs_words_train:   training data, [n_docs, n_words] numpy array
    docs_words_test:    test data, [n_docs, n_words] numpy array
    """
    data = pd.read_csv(filename, dtype=int, sep=' ', names=['doc', 'word', 'train', 'test'])

    n_docs = np.amax(data.loc[:, 'doc'])
    n_words = np.amax(data.loc[:, 'word'])

    docs_words_train = np.zeros((n_docs, n_words), dtype=int)           #training and testing have the same dimension
    docs_words_test = np.zeros((n_docs, n_words), dtype=int)

    docs_words_train[data.loc[:, 'doc'] - 1, data.loc[:, 'word'] - 1] = data.loc[:, 'train']
    docs_words_test[data.loc[:, 'doc'] - 1, data.loc[:, 'word'] - 1] = data.loc[:, 'test']
    print('train', docs_words_train)
    print('test',docs_words_test)
    return docs_words_train, docs_words_test


def autocor(data, lag):
    '''
    function compute the auto-correlation with lag
    :param data: the input data we need to compute the correlation from
    :param lag:  the lag
    :return:     the auto-correlation with stated lag
    '''

    length = data.shape[0]
    mean = sum(data)/length
    data_copy = data.copy()
    s0 = (np.square(data_copy - mean).sum())/length

    y_1 = data[:length-lag]
    y_2 = data[lag:]
    s_lag = np.matmul((y_1 - mean),(y_2 - mean))/length

    return s_lag/s0


def main():

    print('Running toyexample.data with the standard sampler')

    docs_words_train, docs_words_test = read_data('toyexample.data.txt')    #the train is 6(document) * 6(word), for (i,j), the number is the occurance of jth word
                                                                            # in i th document
    n_docs, n_words = docs_words_train.shape
    n_topics = 3
    alpha = 1
    beta = 1
    random_seed = 0

    '''
    #tuning alpha
    plt.subplots(figsize=(15, 6))
    for beta in [1,5,10]:
        sampler_alpha = GibbsSampler(n_docs=n_docs, n_topics=n_topics, n_words=n_words, alpha=alpha, beta=beta, random_seed=random_seed)
        topic_doc_words_distr, theta, phi = sampler_alpha.run(docs_words_train, docs_words_test, n_iter=1000, save_loglike=True)
        ll_train_alpha, ll_test_alpha = sampler_alpha.get_loglike()
        plt.plot(ll_test_alpha, label="beta = {}".format(beta))
    plt.xlabel("iteration")
    plt.ylabel("log-likelihood of testing standard Gibb ")
    plt.legend(loc="lower right")
    '''
    '''
    #tuning the number of topic
    for n_topics in [2,3,4,5]:
        sample_topic = GibbsSampler(n_docs=n_docs, n_topics=n_topics, n_words=n_words, alpha=alpha, beta=beta, random_seed=random_seed)
        topic_doc_words_distr, theta, phi = sample_topic.run(docs_words_train, docs_words_test, n_iter=1000, save_loglike=True)
        ll_train_topic, ll_test_topic = sample_topic.get_loglike()
        plt.plot(ll_test_topic, label="K = {}".format(n_topics))
    plt.xlabel("iteration")
    plt.ylabel("log_likelihood of testing standard Gibbs")
    plt.legend(loc = "lower right")
    '''




    sampler = GibbsSampler(n_docs=n_docs, n_topics=n_topics, n_words=n_words,
                           alpha=alpha, beta=beta, random_seed=random_seed)

    topic_doc_words_distr, theta, phi = sampler.run(docs_words_train, docs_words_test,
                                                    n_iter=200, save_loglike=True)

    print(phi * [phi > 1e-2])

    like_train, like_test = sampler.get_loglike()

    plt.subplots(figsize=(15, 6))
    plt.plot(like_train, label='train')
    plt.ylabel('loglike')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

    '''
    plt.subplots(figsize=(15, 6))
    plot_set = []
    for lag in range(200):                     #auto plot of train std gibbs, 20 iterations burn-in
        auto = autocor(like_train[30:], lag)
        #plt.scatter(lag,auto, color = "grey")
        plot_set.append(auto)
    plt.plot(plot_set)
    plt.ylim(-0.1,0.1)
    plt.xlabel("lag")
    plt.ylabel("auto-correlation of training standard Gibbs")
    plt.show()
    '''



    plt.subplots(figsize=(15, 6))
    plt.plot(like_test, label='test')
    plt.ylabel('loglike')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

    '''
    plt.subplots(figsize=(15, 6))
    plot_set = []
    for lag in range(200):                     #auto plot of testing std gibbs, 20 iterations burn-in
        auto = autocor(like_test[30:], lag)
        #plt.scatter(lag,auto, color = "grey")
        plot_set.append(auto)
    plt.plot(plot_set)
    plt.ylim(-0.1,0.1)
    plt.xlabel("lag")
    plt.ylabel("auto-correlation of testing standard Gibbs")
    plt.show()
    '''


    print('Running toyexample.data with the collapsed sampler')

    sampler_collapsed = GibbsSamplerCollapsed(n_docs=n_docs, n_topics=n_topics, n_words=n_words,
                                              alpha=alpha, beta=beta, random_seed=random_seed)

    doc_word_samples = sampler_collapsed.run(docs_words_train, docs_words_test,
                                             n_iter=200, save_loglike=True)
    topic_counts = np.zeros((n_topics, 6))
    for doc in range(doc_word_samples.shape[0]):
        for word in range(doc_word_samples.shape[1]):
            for topic in doc_word_samples[doc, word]:
                topic_counts[topic, word] += 1

    print(topic_counts)


    '''
    plt.subplots(figsize=(15, 6))
    for beta in [1,5,10]:                         #tuning alpha
        sampler_alpha = GibbsSamplerCollapsed(n_docs=n_docs, n_topics=n_topics, n_words=n_words, alpha=alpha, beta=beta, random_seed=random_seed)
        _ = sampler_alpha.run(docs_words_train, docs_words_test, n_iter=1000, save_loglike=True)
        ll_train_alpha, ll_test_alpha = sampler_alpha.get_loglike()
        plt.plot(ll_train_alpha, label="beta = {}".format(beta))
    plt.xlabel("iteration")
    plt.ylabel("log-likelihood of train collapsed Gibbs")
    plt.legend(loc="lower right")
    '''

    '''
    #tuning the number of topic
    for n_topics in [2,3,4,5]:
        sample_topic = GibbsSamplerCollapsed(n_docs=n_docs, n_topics=n_topics, n_words=n_words, alpha=alpha, beta=beta, random_seed=random_seed)
        _ = sample_topic.run(docs_words_train, docs_words_test, n_iter=1000, save_loglike=True)
        ll_train_topic, ll_test_topic = sample_topic.get_loglike()
        plt.plot(ll_test_topic, label="K = {}".format(n_topics))
    plt.xlabel("iteration")
    plt.ylabel("log_likelihood of testing collapsed Gibbs")
    plt.legend(loc = "lower right")
    '''




    like_train, like_test = sampler_collapsed.get_loglike()

    plt.subplots(figsize=(15, 6))
    plt.plot(like_train, label='train')
    plt.ylabel('loglike')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

    '''
    plt.subplots(figsize=(15, 6))
    plot_set = []
    for lag in range(200):                     #auto plot of train collapsed gibbs, 15 iterations burn-in
        auto = autocor(like_train[30:], lag)
        #plt.scatter(lag,auto, color = "blue", marker = 'o')
        plot_set.append(auto)
    plt.plot(plot_set)
    plt.ylim(-0.1,0.1)
    plt.xlabel("lag")
    plt.ylabel("auto-correlation of training Collapsed Gibbs")
    plt.show()
    '''




    plt.subplots(figsize=(15, 6))
    plt.plot(like_test, label='test')
    plt.ylabel('loglike')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

    '''
    plt.subplots(figsize=(15, 6))
    plot_set = []
    for lag in range(200):                     #auto plot of test col gibbs, 15 iterations burn-in
        auto = autocor(like_test[30:], lag)
        #plt.scatter(lag,auto, color = "blue", marker = 'o')
        plot_set.append(auto)
    plt.plot(plot_set)
    plt.ylim(-0.1,0.1)
    plt.xlabel("lag")
    plt.ylabel("auto-correlation of testing Collapsed Gibbs")
    plt.show()
    '''
if __name__ == '__main__':
    main()
