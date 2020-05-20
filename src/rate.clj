(ns rate (:gen-class))
(require '[clojure.data.csv :as csv]
         '[clojure.java.io :as io])

(defn log-likelihood-multi
  [win-a rating-a rating-b]
  (let [log-a (* win-a (Math/log rating-a)) 
        divisor (* win-a (Math/log (+ rating-a rating-b)))]
    (- log-a divisor)))

(def log-likelihood-single (partial log-likelihood-multi 1))

(defn print-result [{:keys [player-a player-b result]}]
  (println (str player-a
            (case result
              :win " won against "
              :loss " lost against")
            player-b ".")))

(defn get-all-players [results]
  (-> results
    #(vector (% :player-a) (% :player-b))
    (mapcat)
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

(def square #(* % %))

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
(defn minorize-maximize [x w n mean sd]
  (let [variance (square sd)
        discrim (+ (* 4 w variance) (square (- mean (* n variance))))]
    (max machine-epsilon (/ (+ (- (* n variance)) mean (Math/sqrt discrim)) 2))))

; optimize all ratings once
(defn optimize [setup]
  (let [current-ratings @(setup :ratings)]
    (doseq [[player rating] current-ratings]
      (let [w ((setup :total-wins) player)
            other-players (disj (setup :players) player)
            sum-term #(/ 
                       (get-total-games player % (setup :win-matrix)) 
                       (+ (current-ratings player) (current-ratings %)))
            n (apply + (map sum-term other-players))]
        (swap! (setup :ratings) assoc player (minorize-maximize rating w n (setup :mean) (setup :sd)))))
    (let [ratings-ref @(setup :ratings)]
      {:updated-ratings ratings-ref :delta (merge-with - current-ratings ratings-ref)})))

(defn optimize-until [setup delta-pred]
  (loop [result (optimize setup)]
    (if (delta-pred (result :delta))
      result
      (recur (optimize setup)))))

(def standard-pred (comp (partial every? #(<= % 0.00001)) vals))

(defn read-row [[player-a player-b result]]
  {:player-a player-a :player-b player-b :result 
   (case result
    "win" :win
    "loss" :loss)})

(defn -main [& args]
  (let [csv-file (first args)
        mean (Float/parseFloat (nth args 1 "100"))
        sd (Float/parseFloat (nth args 2 "50"))
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
