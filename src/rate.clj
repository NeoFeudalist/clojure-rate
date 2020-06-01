(ns rate (:gen-class))
(require '[clojure.data.csv :as csv]
         '[clojure.java.io :as io])

(defn log-likelihood-multi [win-a rating-a rating-b]
  (let [log-a (* win-a (Math/log rating-a)) 
        divisor (* win-a (Math/log (+ rating-a rating-b)))]
    (- log-a divisor)))

(def square #(* % %))
(def cube #(* % % %))

(defn lambert-w-approx [z]
  (if (> z Math/E)
    (let [log-z (Math/log z)
          log-log-z (Math/log log-z)]
      (- log-z log-log-z))
   (/ z Math/E)))

(def small-enough-eps 0.00001)
(def elo-const (/ 400 (Math/log 10)))
(def inv-elo-const (/ 1 elo-const))

(defn lambert-w [z]
  (loop [w (lambert-w-approx z)]
    (let [expw-term (Math/exp w)
          we-term (- (* w expw-term) z)
          w1-term (+ w 1)
          w2-term (+ w 2)
          w1dbl-term (* 2 w1-term)
          new-w (- w (/ we-term (- (* expw-term w1-term) (/ (* w2-term we-term) w1dbl-term))))]
       (if (> (Math/abs (- new-w w)) small-enough-eps)
          (recur new-w)
          new-w))))            

(defn to-elo [x]
  (Math/pow 10 (/ x 400)))
                      
(def log-likelihood-single (partial log-likelihood-multi 1))

(defn print-result [{:keys [player-a player-b result]}]
  (println (str player-a
            (case result
              :win " won against "
              :loss " lost against")
            player-b ".")))

(defn get-all-players [results]
  (->> results
    (mapcat #(vector (% :player-a) (% :player-b)))
    (set)))

(defn get-all-players-win-matrix [win-matrix]
  (set (mapcat #(first %) win-matrix)))

(defn init-win-matrix [results]
  (let [mat (atom {})]
    (doseq [{:keys [player-a player-b result]} results]
      (let [pair (case result
                  :win [player-a player-b]
                  :loss [player-b player-a])
            val (@mat pair)]
          (if val
            (swap! mat assoc pair (inc val))
            (swap! mat assoc pair 1))))
    @mat))


(defn log-likelihood-win-matrix [win-matrix ratings]
  (->> win-matrix
    (map (fn [[[player-a player-b] wins]] 
            (let [rating-a (ratings player-a) 
                  rating-b (ratings player-b)]
              (log-likelihood-multi wins rating-a rating-b))))
    (apply +)))  

(defn log-prior [mean sd x]
  (- 0 
   (Math/log sd) 
   (/ (Math/log (* 2 Math/PI)) 2)
   (/ (square (/ (- x mean) sd))) 2))

(defn log-posterior-win-matrix [win-matrix ratings mean sd]
  (+ (log-likelihood-win-matrix win-matrix ratings)
   (apply + (map (partial log-prior mean sd) (vals ratings)))))

(defn init-val [val xs]
  (zipmap xs (take (count xs) (repeat val))))

(def init-zeros (partial init-val 0))

(defn init-zeros-win-matrix [win-matrix]
  (init-zeros (get-all-players-win-matrix win-matrix)))
 
(defn get-total-wins [win-matrix]
  (let [win-map (atom (init-zeros-win-matrix win-matrix))]
    (doseq [[[player-a player-b] wins] win-matrix]
      (swap! win-map update player-a (partial + wins)))
    @win-map))

(def default-zero #(or % 0))

(defn get-total-games [player-a player-b win-matrix]
  (+ (default-zero (win-matrix [player-b player-a])) (default-zero (win-matrix [player-a player-b]))))

; game list triple: {:results [...] :mean ... :sd ...} -> expand with :win-matrix
(defn with-win-matrix [setup]
  (assoc setup :win-matrix (init-win-matrix (setup :results))))

; add a list of all players
(defn init-players [setup]
  (assoc setup :players (get-all-players (setup :results)))) 

; add an atomic ratings list
(defn init-ratings [setup]
  (assoc setup :ratings (atom (init-val (setup :mean) (setup :players)))))

; add wins and total games for optimizaiton
(defn add-total-wins [setup]
  (assoc setup :total-wins (get-total-wins (setup :win-matrix))))

; add everything above
(defn add-everything [setup]
  (-> setup (with-win-matrix) (init-players) (init-ratings) (add-total-wins)))

(def machine-epsilon (Math/ulp 1.0))

; use a minorized version of the log-posterior to make it easier to optimize.
; then we solve a quadratic equation to optimize it
(defn minorize-maximize [w n mean sd]
  (let [sd2 (square sd)
        c elo-const
        ic1 inv-elo-const
        ic2 (square ic1)
        term1 (/ (* w sd2) c)
        exp-factor (Math/exp (+ (* ic2 sd2 w) (* ic1 mean)))
        lambert-arg (* ic2 sd2 n exp-factor)
        term2 (- (* c (lambert-w lambert-arg)))]
    (+ mean term1 term2)))

; optimize all ratings once
(defn optimize [setup]
  (let [current-ratings @(setup :ratings)]
    (doseq [[player rating] current-ratings]
      (let [w ((setup :total-wins) player)
            other-players (disj (setup :players) player)
            sum-term #(/ 
                       (get-total-games player % (setup :win-matrix)) 
                       (+ (to-elo (current-ratings player)) (to-elo (current-ratings %))))
            n (apply + (map sum-term other-players))]
        (swap! (setup :ratings) assoc player (minorize-maximize w n (setup :mean) (setup :sd)))))
    (let [ratings-ref @(setup :ratings)]
      {:updated-ratings ratings-ref :delta (merge-with - current-ratings ratings-ref)})))

(defn optimize-until [setup delta-pred]
  (loop [result (optimize setup)]
    (if (delta-pred (result :delta))
      result
      (recur (optimize setup)))))

(def standard-pred (comp (partial every? #(<= % small-enough-eps)) vals))

(defn read-row [[player-a player-b result]]
  {:player-a player-a :player-b player-b :result 
   (case result
    "win" :win
    "loss" :loss)})

(defn -main [& args]
  (let [csv-file (first args)
        mean (Float/parseFloat (nth args 1 "1500"))
        sd (Float/parseFloat (nth args 2 "350"))
        results-name (str "results-" csv-file)]
    ; read csv file
    (with-open [reader (io/reader csv-file)
                writer (io/writer results-name)]
      (let [results (->> reader
                     (csv/read-csv)
                     (mapv read-row))
            triple {:results results :mean mean :sd sd}
            with-everything (add-everything triple)]
        (optimize-until with-everything standard-pred)              
        (csv/write-csv writer (vec @(with-everything :ratings)))))
    (println "Optimization done, ratings written to" (str results-name "."))))
