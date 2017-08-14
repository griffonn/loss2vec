import tensorflow as tf
from word2vec_dlce
# from word2vec_dlce import Options
# from word2vec_dlce import Word2Vec


def eval(model):
  """Evaluate analogy questions and reports accuracy."""

  # How many questions we get right at precision@1.
  correct = 0

  try:
    total = model._analogy_questions.shape[0]
  except AttributeError as e:
    raise AttributeError("Need to read analogy questions.")

  failed_test_idx = []
  pass_test_idx = []

  start = 0
  while start < total:
    limit = start + 2500
    sub = model._analogy_questions[start:limit, :]
    idx = model._predict(sub)
    start = limit
    for i,question in enumrate(xrange(sub.shape[0])):
      for j in xrange(4):
        if idx[question, j] == sub[question, 3]:
          pass_test_idx.append(i + (start-2500))

          # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
          correct += 1
          break
        elif idx[question, j] in sub[question, :3]:
          # We need to skip words already in the question.
          continue
        else:
          failed_test_idx.append(i + (start-2500))
          # The correct label is not the precision@1
          break
  return correct, total, failed_test_idx, pass_test_idx


def load_session(path):
  sess=tf.Session()    
  #First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph('my_test_model-1000.meta')
  return saver.restore(sess,tf.train.latest_checkpoint('./'))

if __name__ == "__main__":

  MODEL_DLCE_PATH = 
  MODEL_PATH = '/cs/engproj/3deception/meaning/models_data/text8/model.ckpt-37152.index'

  opts = Options()
  sess = load_model(MODEL_PATH)
  sess_dlce = load_model(MODEL_DLCE_PATH)

  with tf.device("/cpu:0"):
    model = Word2Vec(opts, sess)
    model.read_analogies() # Read analogy questions    

    model_dlce = Word2Vec(opts, sess_dlce)
    model_dlce.read_analogies() # Read analogy questions
  
    correct, total, failed_test_idx, pass_test_idx = eval(model)  # Eval analogies.    
    correct_d, total_d, failed_test_idx_d, pass_test_idx_d = eval(model_dlce)  # Eval analogies.
  print("Original Results:")
  print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))
  print("Eval %4d/%d accuracy = %4.1f%%" % (correct_d, total_d, correct_d * 100.0 / total_d))

  both_failed = list(set(failed_test_idx).intersection(failed_test_idx_d))
  both_sucess = list(set(pass_test_idx).intersection(pass_test_idx_d))
  reg_sucess = [x for x in pass_test_idx if x not in both_sucess]
  dLCE_sucess = [x for x in pass_test_idx_d if x not in both_sucess]

  print("Removing {} examples normal model successed on but dLCE model failed".format(len(reg_sucess)))
  new_test_list_idx = sorted(both_failed + dLCE_sucess + both_sucess)

  with open(TESTS_PATH, 'rb') as tests:
    tests_content = tests.readlines()
  
  with open('NEW_TESTS.txt','w') as new_tests:
    for i,line in enumrate(tests_content):
      if i in new_test_list_idx:
        new_tests.write(line)

  print("Finished Script NEW_TESTS.txt saved")

