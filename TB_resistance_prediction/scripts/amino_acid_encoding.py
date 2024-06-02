



class AminoAcidEncoder(): 

    def __init__(self, RIF_sequence_path): 
        self.RIF_sequence_path = RIF_sequence_path 

        self.sequences = self.read_data_from_file(self.RIF_sequence_path)

    def read_data_from_file(self, data_file):
        '''
        return RIF sequences and train strains data into a list
        '''
        data = []
        with open(data_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    ID, gene, sequence = parts
                    data.append((ID, gene, sequence))
                if len(parts) == 2:
                    ID, resistance = parts
                    data.append((ID, resistance))
        return data


    def translate(self, seq):

        codon_table = { 
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
        'TAC':'Y', 'TAT':'Y', 'TAA':'<mask>', 'TAG':'<mask>', 
        'TGC':'C', 'TGT':'C', 'TGA':'<mask>', 'TGG':'W', }

        acceptable_chars = ['A','T','C','G', 'X']
        validation = [i in acceptable_chars for i in seq]

        protein =""

        #if all(validation):
        if seq == 'None':
            protein = 'None'
        else :
            if len(seq)%3 == 0:
                for i in range(0, len(seq), 3):
                    codon = seq[i:i + 3]
                    #print(codon)
                    protein += codon_table[codon]

            if len(seq)%3 == 1:
                for i in range(0, len(seq)-1, 3):
                    codon = seq[i:i + 3]
                    #print(codon)
                    protein += codon_table[codon]

            if len(seq)%3 == 2:
                for i in range(0, len(seq)-2, 3):
                    codon = seq[i:i + 3]
                    #print(codon)
                    protein += codon_table[codon]

        return protein

    def encoding_DNA_to_aa(self):

        aa_sequences = []

        for idx, line in enumerate(self.sequences) : 
            ID = line[0]
            gene = line[1]
            DNA_sequence = line[2] 

            aa_sequence = self.translate(DNA_sequence)

            row = f"{ID} {gene} {aa_sequence}"
            aa_sequences.append(row)

        # store AA sequences into a txt file
        with open("/home/thm333/TB_resistance_prediction_2/data/aa_RIF.txt", "w") as file:
            # Write each row to the file, separated by newlines
            for row in aa_sequences:
                file.write(row + "\n")



## TEST THE CLASS 
RIF_sequence_file = '/home/thm333/TB_resistance_prediction_2/data/sequences_RIF'

AA_DNA_class = AminoAcidEncoder(RIF_sequence_path=RIF_sequence_file)
AA_DNA_class.encoding_DNA_to_aa()






