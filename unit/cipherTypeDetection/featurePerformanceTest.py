import unittest
import timeit
from cipherTypeDetection import cipherStatisticsDataset as ds
import unit.cipherImplementations.cipherTest as cipherTest
from util.utils import map_text_into_numberspace
import numpy as np


class FeaturePerformanceTest(unittest.TestCase):
    cipher = cipherTest.CipherTest.cipher
    ALPHABET = map_text_into_numberspace(cipher.alphabet, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    plaintext = b'thisisafilteredplaintextwithsomewordsinittobeusedtotestthestatisticsofthetextlinetocipherstatisticsd'
    plaintext_numberspace = map_text_into_numberspace(plaintext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    plaintext_numberspace = [int(d) for d in plaintext_numberspace]
    # key = fcghartokldibuezjpqxyvwnsm
    # cipher: simple substitution
    ciphertext = b'xokqkqfrkixapahzifkuxanxwkxoqebawephqkukxxecayqahxexaqxxoaqxfxkqxkgqerxoaxanxikuaxegkzoapqxfxkqxkgqh'
    ciphertext_numberspace = map_text_into_numberspace(ciphertext, cipher.alphabet, cipherTest.CipherTest.UNKNOWN_SYMBOL_NUMBER)
    ciphertext_numberspace = [int(d) for d in ciphertext_numberspace]

    def test01calculate_frequencies(self):
        # unigrams
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_frequencies(self.ciphertext_numberspace, 1, recursive=False), number=10000)
        print(t / 10)

        # bigrams
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_frequencies(self.ciphertext_numberspace, 2, recursive=False), number=10000)
        print(t / 10)

        # trigrams
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_frequencies(self.ciphertext_numberspace, 3, recursive=False), number=1000)
        print(t / 10)

        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_frequencies(self.ciphertext_numberspace, 3, recursive=True), number=1000)
        print(t / 10)

    def test02calculate_ny_gram_frequencies(self):
        # bigrams interval=2
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 2, interval=2, recursive=False),
                               number=10000)
        print(t / 10)

        # bigrams interval=7
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 2, interval=7, recursive=False),
                               number=10000)
        print(t / 10)

        # trigrams interval=7
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 3, interval=7, recursive=False),
                               number=1000)
        print(t / 10)

        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_ny_gram_frequencies(self.ciphertext_numberspace, 3, interval=7, recursive=True),
                               number=1000)
        print(t / 10)

    def test03calculate_index_of_coincidence(self):
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_index_of_coincidence(self.ciphertext_numberspace), number=100000)
        print(t / 10)

    def test04calculate_index_of_coincidence_bigrams(self):
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_digraphic_index_of_coincidence(self.ciphertext_numberspace), number=10000)
        print(t / 10)

    def test05has_letter_j(self):
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.has_letter_j(self.ciphertext_numberspace), number=100000)
        print(t / 10)

    def test06calculate_chi_square(self):
        unigram_frequencies = [0]*26
        for c in self.cipher.alphabet:
            for i in range(0, len(self.ciphertext)):
                if self.ciphertext[i] == c:
                    unigram_frequencies[self.cipher.alphabet.index(c)] += 1
            unigram_frequencies[self.cipher.alphabet.index(c)] = unigram_frequencies[self.cipher.alphabet.index(c)] / len(self.ciphertext)
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_chi_square(unigram_frequencies), number=10000)
        print(t / 10)

    def test07pattern_repetitions(self):
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.pattern_repetitions(self.ciphertext_numberspace), number=10000)
        print(t / 10)

    def test08calculate_entropy(self):
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_entropy(self.ciphertext_numberspace), number=10000)
        print(t / 10)

    def test09calculate_autocorrelation_average(self):
        ciphertext_numberspace = np.array(self.ciphertext_numberspace)
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_autocorrelation(ciphertext_numberspace), number=10000)
        print(t / 10)

    def test10calculate_sdd(self):
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_sdd(self.ciphertext_numberspace), number=10000)
        print(t / 10)

    def test11_calculate_ldi_stats(self):
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_ldi_stats(self.ciphertext_numberspace), number=10)
        print(t / 10)

    def test12_calculate_cdd(self):
        t = 0
        for _ in range(10):
            t += timeit.timeit(lambda: ds.calculate_cdd(self.ciphertext_numberspace), number=10000)
        print(t / 10)
    '''The methods calculate_statistics and encrypt can not be tested properly, because they are either random or are only depending on
    other methods'''
