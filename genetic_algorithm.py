import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

#  بارگذاری داده‌های پیش‌ پردازش‌ شده
X_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')

# یک شبکه عصبی کوچک برای محاسبه میزان دقت با ماسک(کروموزوم)انتخاب شده
def little_mlp(X_selected, y):

    #به داده های تست و آموزشX_selectedتقسیم
    X_tr, X_val, y_tr, y_val = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    #ساخت شی و آموزش مدل
    mlp = MLPClassifier(
        hidden_layer_sizes=(16,),
        max_iter=100,
        early_stopping=True,
        random_state=42
)

    mlp.fit(X_tr, y_tr)

    #تست مدل
    prediction = mlp.predict(X_val)
    accuracy = accuracy_score(y_val, prediction)

    return accuracy

# تابع برازش
#نود درصد همیت دقت + ده درصد اهمیت خلوت بودن فیچرها(برای سرعت بالاتر)
def calculate_fitness(chromosome, X, y):

    #اگر تمام اعضای یک کروموزم صفر بودند حذف می شود
    if np.sum(chromosome) == 0:
        return 0.0
    #عملیات ماسک گذاری(انتخاب فقط ستون‌هایی که ژن آن‌ها برابر 1 است)
    X_selected = X[:, chromosome == 1]

    #محاسبه دقت
    accuracy = little_mlp(X_selected, y)

    #محاسبه امتیاز نهایی
    num_feature_selected = np.sum(chromosome)
    total_feachers = len(chromosome)
    fittness = (0.9 * accuracy) + (0.1 * (1.0 - (num_feature_selected / total_feachers)))

    return fittness

#Roulette Wheel Selection
def select_parents(population, fittness_scores):

    #محاسبه مجموع امتیازها و شانس هر کروموزوم
    scores_sumation = np.sum(fittness_scores)

    if scores_sumation == 0:
        #به همه شانس یکسان تعلق می گیرد
        chances = np.ones(len(population)) / len(population)
    else:
        chances = fittness_scores / scores_sumation
    
    #چرخاندن رول و انتخاب والدین
    selected_indices = np.random.choice(len(population), size=2, p=chances)
    # اگر هر دو ایندکس یکی بودند، دومی را دوباره انتخاب کن
    while selected_indices[0] == selected_indices[1]:
        selected_indices[1] = np.random.choice(len(population), p=chances)

    parent1 = population[selected_indices[0]]
    parent2 = population[selected_indices[1]]

    return parent1,parent2

#Single-Point Crossover
def Crossover(parent1,parent2):
    #انتخاب نقطه برش
    point = np.random.randint(1, len(parent1))
    
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    
    return child1, child2

def mutate(chromosome, mutation_rate = 0.01):

    for i in range(len(chromosome)):
        # تولید یک عدد تصادفی بین 0 و 1
        mutation_chance = np.random.rand()
        if mutation_chance <= mutation_rate:
            chromosome[i] = 1 - chromosome[i]
        
    return chromosome

def genetic_algorithm(X_train, y_train, population, num_generation):

    overall_best_chrom = None
    overall_best_score = -1

    for gen in range(0, num_generation):

        #محاسبه امتیاز برازش برای تمامی کروموزم های جمعیت
        fitness_score = np.array([calculate_fitness(chrom, X_train, y_train) for chrom in population])

        #پیدا کردن بهترین کروموزم با بالاترین امتیاز برازش در این نسل
        best_idx = np.argmax(fitness_score)
        best_chrom = population[best_idx]
        best_score = fitness_score[best_idx]
        print(f"Generation {gen} |Best Score = {best_score}")

        #پیدا کردن بهترین تمام دوران
        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best_chrom = best_chrom.copy()

        new_population = []
        #واردکردن نخبگان به جمعیت جدید
        new_population.append(overall_best_chrom.copy())
        if not np.array_equal(overall_best_chrom, best_chrom):
            new_population.append(best_chrom.copy())

        while len(new_population) < len(population):
            #انتخاب والدین
            parent1, parent2 = select_parents(population, fitness_score)
            #تولید فرزند
            child1, child2 = Crossover(parent1, parent2)
            #جهش برخی ازفرزندان
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < len(population):
                new_population.append(child2)

        population = np.array(new_population)
    
    return population, overall_best_chrom

#تنظیمات اولیه الگوریتم ژنتیک
POP_SIZE = 50       # تعداد افراد جامعه
GENE_LENGTH = X_train.shape[1]   # طول هر کروموزوم (تعداد ویژگی‌ها)

#تولید جمعیت اولیه تصادفی (می توان از حلقه تو در تو هم استفاده کرد که خب هم کندتره هم کثیف تر)
population = np.random.randint(0, 2, size=(POP_SIZE, GENE_LENGTH))

NUM_GENERATIONS = 10
final_population, best_chromosome = genetic_algorithm(X_train, y_train, population, NUM_GENERATIONS)

# شمارش و چاپ تعداد ویژگی‌های انتخاب شده در پایان کار
selected_features_count = np.sum(best_chromosome)
print(f"Total features selected: {selected_features_count} out of {GENE_LENGTH}")

np.save("best_genetic_features.npy", best_chromosome)