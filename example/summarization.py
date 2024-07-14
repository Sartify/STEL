# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:49:19 2024

@author: michael.mollel@sartify.com
"""
import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.cluster import KMeansClusterer
import numpy as np
nltk.download('punkt')

model_name ="sartifyllc/MultiLinguSwahili-bge-small-en-v1.5-nli-matryoshka" 
matryoshka_dims = 64
	
model = SentenceTransformer(model_name, truncate_dim=matryoshka_dims)



article= '''
Mimba za utotoni Katavi zamkera samia, ataka watoto waachwe wakue Jumapili, Julai 14, 2024. Dar es Salaam. Rais Samia Suluhu Hassan amewataka wakazi wa Mkoa wa Katavi kuwaacha watoto wasome, wakue ili kuwaepusha na zahama ya kuzaa watoto wenzao. Ametoa kauli hiyo baada ya kutembelea Hospitali ya Rufaa Mkoa wa Katavi, alipokuwa akikagua maendeleo ya ujenzi wake ambapo pia alitembeka wodi mbalimbali ikiwemo ya watoto njiti. Akizungumza leo Jumapili, Julai 14, 2024 baada ya kukagua maendeleo ya hospitali hiyo, Samia amesema katika wodi ya watoto njiti aliyoitembelea amekuta mwanamke mmoja pekee mwenye umri mkubwa ndiye aliejifungua mtoto ambaye hajatimia uzito, lakini wengine wote ni kati ya miaka 15 hadi 19."Hakuna miaka 20 ni mmoja tu ana 25 mle ndani, 15 hadi 19, hawa ndiyo wanajifungua watoto hawajatimia uzito, wanajifungua wanakwenda nyumbani akirudi mtoto ana matatizo, niombe sana waacheni watoto wakue kidogo angalau mtoto ajifungue kuanzia miaka 19 na kupanda juu 20 lakini siyo 14, 18 au 17," amesema Samia. Amesema umri chini ya miaka 18 ni watoto wadogo, huku akiwataka waachwe waende shule, wakue na waache kuzaa watoto wenzao. "Ujumbe huu ni kwenu ninyi wananchi," amesema Samia. Utafiti wa Afya ya Uzazi na Mtoto pamoja na Viashiria vya Malaria nchini kwa mwaka 2022 (TDHS-MIS) inautaja mkoa wa Katavi miongoni mwa mitano inayoongoza kwa mimba za utotoni. Mkoa Katavi unashika namba tatu kitaifa ukiwa na asilimia 34 ukiwa nyuma ya Songwe wenye asilimia 45, Ruvuma asilimia 37 huku nyuma yake kukiwa na mkoa wa Mara asilimia 31 na Rukwa wenye asilimia 30. Mbali na suala la mimba za utotoni, Samia ametumia nafasi hiyo kuwataka wakazi wa Katavi, #kuunga mkono huduma za bima ya afya ili afya za Watanzania ziiimarike na kila mtu aweze kunufaika nazo. Amesema matibabu ni ghali huku akiwaeka bayana kuwa Mfuko wa Bima ya Afya utakapowatembelea kila mwananchi anunue bima yake na familia yake. Amesema bima ya afya maana yake vifaa vyote, matibabu yote yanapatikana na kuondoa ulazima wa wao kwenda mbali kutafuta huduma."Unakata bima yako mara moja kwa mwaka, unatibiwa na familia yako mwaka mzima kwa kutumia kipande chako cha bima ya afya ili tuweze kujenga na kutoa huduma kubwa zaidi katika hospitali za namna hii, na hata hospitali zetu za mikoa na wilaya na hata zile za afya kule chini," amesema Samia. Samia amesema Serikali itaendelea kujenga majengo mengine ya kutoa huduma na itakamilisha hospitali hiyo kadri ilivyokusudiwa, hasa baada ya kuelezwa kuwa majengo yanayotumika ni yale yaliyojengwa awamu ya kwanza. Amesema jitihada hizo zinalenga kusogeza huduma kwa wananchi, kwani hospitali kama hiyo ilizoeleka kusikika Arusha, Mwanza na Dar es Salaam. Amesema hilo amelishuhudia hata alipotembelea wodi ya watoto waliozaliwa kabla ya wakati (njiti) inafanana na ile iliyopo Dar es Salaam, kwani inavyo vifaa na watoa huduma wanaotakiwa. "Ombi langu kwa watumishi wa hospitali hii ni kuendeleza utoaji wa huduma kama mnapokula kiapo chenu mnapopewa vyeti, niseme tu Serikali imedhamiria kusogeza huduma za afya, hii ni hospitali ya mkoa lakini ndani ya mkoa huu tumefanya kazi ya kuongeza vituo vya afya, hospitali za wilaya, vituo katika kata na zahanati." Amesema hospitali hiyo imempa faraja kwani wakati anaikagua amekutana na madaktari vinara ambao wangependa 'kudunda' Dar es Salaam na miji mingine kama Arusha na Mwanza, lakini wamekubali kukaa Katavi na kuhudumia wananchi. "Niwaoambe Wanakatavi tulinde mali yetu, hii ni mali yetu mnapokuja kufuata huduma fuateni masharti mnayopewa hapa ndani," amesema Samia. Kwa upande wa Naibu Waziri wa Afya, Dk Godwin Mollel amesema tayari Sh4 bilioni za ujenzi zimeshatolewa.
'''

sentences=nltk.sent_tokenize(article)
 
# strip leading and trailing spaces
sentences = [sentence.strip() for sentence in sentences]


data = pd.DataFrame(sentences)
 
data.columns=['sentence']

def get_sentence_embeddings(sentence):
    embedding = model.encode([sentence])
    return embedding[0]

	
data['embeddings']=data['sentence'].apply(get_sentence_embeddings)

NUM_CLUSTERS=10
 
iterations=25
 
X = np.array(data['embeddings'].tolist())
 
kclusterer = KMeansClusterer(
        NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,
        repeats=iterations,avoid_empty_clusters=True)
 
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)


data['cluster']=pd.Series(assigned_clusters, index=data.index)
data['centroid']=data['cluster'].apply(lambda x: kclusterer.means()[x])


from scipy.spatial import distance_matrix
def distance_from_centroid(row):
    #type of emb and centroid is different, hence using tolist below
    return distance_matrix([row['embeddings']], [row['centroid'].tolist()])[0][0]
 
data['distance_from_centroid'] = data.apply(distance_from_centroid, axis=1)

summary=' '.join(data.sort_values('distance_from_centroid',ascending = True).groupby('cluster').head(1).sort_index()['sentence'].tolist())
