(ns dtypemat.bufferimpl
  "Namespace for core.matrix implementation using dtype-next buffers.

   Array format is defined as:
   - Top level object is an instance of clojure.lang.IPersistentVector
   - If the array is 1-dimensional each element is a scalar
   - Otherwise each element is an sub-array with identical shape (1 dimensional or more)

   Note that this allows for other array implementations to be nested inside persistent vectors,
   provided all nested arrays have the same shape and dimensionality of at least 1."
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix.implementations :as imp]
            [clojure.core.matrix.impl.common :refer [mapmatrix]]
            [clojure.core.matrix.impl.mathsops :as mops]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
  #?@(:clj [[clojure.core.matrix.macros :refer [scalar-coerce error doseq-indexed]]
            [clojure.core.matrix.macros-clj :refer [native-array?]]]))

  #?(:clj (:import [clojure.lang IPersistentVector Indexed]
                   [java.util List]
                   [tech.v3.datatype NDBuffer])
     :cljs (:require-macros
              [clojure.core.matrix.impl.persistent-vector :refer [vector-1d?]]
              [clojure.core.matrix.macros :refer [scalar-coerce error doseq-indexed]]
              [clojure.core.matrix.macros-cljs :refer [native-array?]])))

#?(:clj (do
  (set! *warn-on-reflection* true)
  (set! *unchecked-math* true)
))

;; (set! *unchecked-math* :warn-on-boxed) ;; use to check for boxing, some is unavoidable


;; =======================================================================
;; Implementation for dtype.next NDBuffers used as matrices


(extend-protocol mp/PDimensionInfo
;;  #?(:clj IPersistentVector :cljs PersistentVector)
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (dimensionality [m]  (count (dtype/shape m))) ;;TOM check...
  (is-vector? [m]      (== (count (dtype/shape m)) 1))
  (is-scalar? [m] false)
  (get-shape [m] (dtype/shape m))
  (dimension-count [m x] (nth (dtype/shape m) x)))

(extend-protocol mp/PElementCount
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (element-count [m] (dtype/ecount m)))

(extend-protocol mp/PImplementation
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (implementation-key [m] :dtype.next/ndbuffer)
    (meta-info [m]
      {:doc "Implementation for dtype.next buffers used as matrices."})
    (new-vector [m length]       (dtt/native-tensor [length]))
    (new-matrix [m rows columns] (dtt/native-tensor [rows columns]))
    (new-matrix-nd [m dims]
      (if-let [dims (seq dims)]
        (dtt/native-tensor dims)
        0.0))
    (construct-matrix [m data]
      (dtt/ensure-native (dtt/->tensor data :datatype :float64)))
    (supports-dimensionality? [m dims]
      true))

(extend-protocol mp/PBroadcast
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (broadcast [m target-shape]
      (dtt/broadcast m target-shape)))

(extend-protocol mp/PBroadcastLike
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (broadcast-like [m a]
      (mp/broadcast a (mp/get-shape m))))

#_
(defn persistent-vector-coerce
  "Coerces to nested persistent vectors"
  [x]
  (let [dims (long (mp/dimensionality x))]
    (cond
      (> dims 0) (mp/convert-to-nested-vectors x) ;; any array with 1 or more dimensions
      (and (== dims 0) (not (mp/is-scalar? x))) (mp/get-0d x) ;; array with zero dimensionality

      ;; it's not an array - so try alternative coercions
      (nil? x) x
      #?@(:clj [(.isArray (class x)) (mapv persistent-vector-coerce (seq x))])
      #?@(:clj [(instance? List x) (coerce-nested x)])
      (instance? #?(:clj Iterable :cljs IIterable) x) (coerce-nested x)
      (sequential? x) (coerce-nested x)

      ;; treat as a scalar value
      :default x)))

;;waiting....
(extend-protocol mp/PBroadcastCoerce
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (broadcast-coerce [m a]
    (if  (dtt/tensor? a)
        (mp/broadcast a    (dtype/shape m))
        (let [donor-type   (dtype/get-datatype m)]
          (mp/broadcast (dtt/->tensor a :datatype donor-type) (dtype/shape m))))))

(extend-protocol mp/PIndexedAccess
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (get-1d [m x]    (dtt/mget m x))
  (get-2d [m x y] (dtt/mget m x y))
  (get-nd [m indexes] (apply dtt/mget m indexes)))

;; we extend this so that nested mutable implementions are possible
(extend-protocol mp/PIndexedSetting
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (set-1d [m row v]        (dtt/mset! m row v))
    (set-2d [m row column v] (dtt/mset!  m row column v))
    (set-nd [m indexes v]    (apply dtt/mset! m indexes v))
    (is-mutable? [m]
      ;; assume persistent vectors are immutable, even if they may have mutable components
      true))

(extend-protocol mp/PMatrixSlices
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (get-row    [m i]      (m i))
    (get-column [m i]      ((dtt/columns m) i))
    (get-major-slice [m i] (m i))
    (get-slice [m dimension i] (m dimension i) ;;TOM uncertain...
      ))

(extend-protocol mp/PMatrixRows
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
	  (get-rows [m] (dtt/rows m)))

(extend-protocol mp/PMatrixColumns
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
	  (get-columns [m] (dtt/columns m)))

(extend-protocol mp/PSliceView
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (get-major-slice-view [m i] (m i)))

(extend-protocol mp/PSliceView2
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (get-slice-view [m dimension i]
      ;; delegate to get-slice
      (mp/get-slice m dimension i)))

(extend-protocol mp/PSliceSeq
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (get-major-slice-seq [m]
      (dtt/rows m)))

(extend-protocol mp/PSliceJoin
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (join [m a]
      (let [dims  (mp/dimensionality m)
            adims (mp/dimensionality a)
            shp   (dtype/shape m)
            shp2  (dtype/shape a)
            n1    (shp  0)
            n2    (shp2 0)]
        (cond
          (== dims adims)
          (dtt/construct-tensor (dtype/concat-buffers [(dtype/as-buffer m) (dtype/as-buffer a)])
                                (tech.v3.tensor.dimensions/dimensions (assoc shp 0 (+ n1 n2))))
          (== dims (inc adims))
          (dtt/construct-tensor (dtype/concat-buffers [(dtype/as-buffer m) (dtype/as-buffer a)])
                                (tech.v3.tensor.dimensions/dimensions (assoc shp 0 (inc n1))))
          :else
            (error "Joining with array of incompatible size")))))

;;PENDING
(extend-protocol mp/PRotate
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (rotate [m dim places]
      (let [dim (long dim)
            places (long places)]
        (if (== 0 dim)
         (let [c (long (count m))
               sh (long (if (> c 0) (mod places c) 0))]
           (if (== sh 0)
             m
             (vec (concat (subvec m sh c) (subvec m 0 sh)))))
         (mapv (fn [s] (mp/rotate s (dec dim) places)) m)))))

(extend-protocol mp/PTransposeDims
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (transpose-dims [m ordering]
      (if-let [ordering (seq ordering)]
        (let [dim (long (first ordering))
              next-ordering (map (fn [i] (if (< i dim) i (dec i))) (next ordering))
              slice-range (range (mp/dimension-count m dim))]
          (mapv (fn [si] (mp/transpose-dims (mp/get-slice m dim si) next-ordering)) slice-range))
        m)))

(extend-protocol mp/POrder
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (order
    ([m indices]
      (mapv #(nth m %) (mp/element-seq indices)))
    ([m dimension indices]
      (let [dimension (long dimension)]
        (if (== dimension 0)
          (mp/order m indices)
          (mapv #(mp/order % (dec dimension) indices) m))))))

(extend-protocol mp/PSubVector
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (subvector [m start length]
      (subvec m start (+ (long start) (long length)))))

(extend-protocol mp/PValidateShape
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (validate-shape 
      ([m]
        (if (mp/same-shapes? m)
          (mp/get-shape m)
          (error "Inconsistent shape for persistent vector array.")))
      ([m shape]
        (when (empty? shape) (error "Expected empty shape for persistent vector: " m)) 
        (if (apply = (next shape) (map mp/validate-shape m))
            shape
            (error "Inconsistent shape for persistent vector array, expected: " shape " on array " m)))))

(extend-protocol mp/PMatrixAdd
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (matrix-add [m a]
      (let [[m a] (mp/broadcast-compatible m a)]
        (mapmatrix + m (persistent-vector-coerce a))))
    (matrix-sub [m a]
      (let [[m a] (mp/broadcast-compatible m a)]
        (mapmatrix - m (persistent-vector-coerce a)))))

(extend-protocol mp/PVectorOps
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (vector-dot [a b]
      ;; optimised vector-dot for persistent vectors, handling 1D case
      (let [dims (long (mp/dimensionality b))]
        (cond
          (and (== dims 1) (== 1 (long (mp/dimensionality a))))
            (let [n (long (count a))]
              (cond
                (not= n (long (long (mp/dimension-count b 0)))) (error "Mismatched vector sizes")
                (instance? List b)
                  (let [b ^List b]
                    (loop [i 0 res 0.0]
                      (if (>= i n)
                        res
                        (recur (inc i) (+ res (* (double (nth a (int i))) (double (.get b (int i)))))))))
                (native-array? ^Object b)
                  (loop [i 0 res 0.0]
                    (if (>= i n)
                      res
                      (recur (inc i) (+ res (* (double (nth a (int i))) (double (nth b i)))))))
                :else
                  (reduce + (map * a (mp/element-seq b)))))

          :else (mp/inner-product a b))))
    (length [a]
      (if (number? (first a))
        (let [n (long (count a))]
         (loop [i 0 res 0.0]
           (if (< i n)
             (let [x (double (nth a i))]
               (recur (inc i) (+ res (* x x))))
             (Math/sqrt res))))
        (Math/sqrt (mp/length-squared a))))
    (length-squared [a]
      (if (number? (first a)) 
        (let [n (long (count a))]
          (loop [i 0 res 0.0]
            (if (< i n)
              (let [x (double (nth a i))]
                (recur (inc i) (+ res (* x x))))
              res)))
        (mp/element-reduce a (fn [^double r ^double x] (+ r (* x x))) 0.0)))
    (normalise [a]
      (mp/scale a (/ 1.0 (Math/sqrt (mp/length-squared a))))))

(extend-protocol mp/PMutableMatrixConstruction
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (mutable-matrix [m]
      nil ;; fall-though: should get an ndarray result
      ))

(extend-protocol mp/PImmutableMatrixConstruction
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (immutable-matrix [m]
    m))

(extend-protocol mp/PVectorDistance
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (distance [a b] (mp/length (mp/matrix-sub b a))))

(extend-protocol mp/PSummable
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (element-sum [a]
      (mp/element-reduce a +)))

(extend-protocol mp/PCoercion
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (coerce-param [m param]
      (persistent-vector-coerce param)))

(extend-protocol mp/PMatrixEquality
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (matrix-equals [a b]
      (let [bdims (long (mp/dimensionality b))
            acount (long (count a))]
        (cond
          (<= bdims 0)
            false
          (not= acount (mp/dimension-count b 0))
            false
          (== 1 bdims)
            (and (== 1 (long (mp/dimensionality a)))
                 (loop [i 0]
                   (if (< i acount)
                     (if (mp/matrix-equals (nth a i) (mp/get-1d b i)) ;; can't avoid boxed warning, may be any sort of number
                       (recur (inc i))
                       false)
                     true)))
          (vector? b)
            (let [n acount]
               (loop [i 0]
                     (if (< i n)
                       (if (mp/matrix-equals (nth a i) (b i))
                         (recur (inc i))
                         false)
                       true)))
          :else
            (loop [sa (seq a) 
                   sb (mp/get-major-slice-seq b)]
              (if sa
                (if (mp/matrix-equals (first sa) (first sb))
                  (recur (next sa) (next sb))
                  false)
                true))))))

(extend-protocol mp/PMatrixMultiply
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (element-multiply [m a]
      (if (number? a)
        (mp/scale m a)
        (let [[m a] (mp/broadcast-compatible m a)]
          (mp/element-map m * a))))
    (matrix-multiply [m a]
      (let [mdims (long (mp/dimensionality m))
            adims (long (mp/dimensionality a))]
        (cond
          (== adims 0) (mp/scale m a)
          (and (== mdims 1) (== adims 2))
            (vec (for [i (range (mp/dimension-count a 1))]
                     (let [r (mp/get-column a i)]
                       (mp/vector-dot m r))))
          (and (== mdims 1) (== adims 1))
            (mp/vector-dot m a)
          (and (== mdims 2) (== adims 1))
            (mapv #(mp/vector-dot % a) m)
          (and (== mdims 2) (== adims 2))
            (mapv (fn [r]
                     (vec (for [j (range (mp/dimension-count a 1))]
                            (mp/vector-dot r (mp/get-column a j))))) m)
          :else
            (mp/inner-product m a)))))

(extend-protocol mp/PMatrixProducts
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (inner-product [m a]
      (let [adims (long (mp/dimensionality a))
            mdims (long (mp/dimensionality m))]
        (cond
          (== 0 adims)
            (mp/scale m (mp/get-0d a))
          (not (== (long (last (mp/get-shape m))) (long (first (mp/get-shape a)))))
            (error "Incompatible shapes for inner product: " (mp/get-shape m) " and " (mp/get-shape a))
          (== 1 mdims)
            (if (== 1 adims)
              (mp/element-sum (mp/element-multiply m a))
              (reduce mp/matrix-add (map (fn [sl x] (mp/scale sl x))
                                       (mp/get-major-slice-seq a)
                                       (mp/get-major-slice-seq m)))) ;; TODO: implement with mutable accumulation
          :else
           (mapv #(mp/inner-product % a) (mp/get-major-slice-seq m)))))
    (outer-product [m a]
      (mp/element-map m (fn [v] (mp/pre-scale a v)))))

(extend-protocol mp/PVectorTransform
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (vector-transform [m a]
      (mp/matrix-multiply m a))
    (vector-transform! [m a]
      (mp/assign! a (mp/matrix-multiply m a))))

(extend-protocol mp/PMatrixScaling
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (scale [m a]
      (mapmatrix #(* % a) m)) ;; can't avoid boxed warning, may be any sort of number
    (pre-scale [m a]
      (mapmatrix #(* a %) m))) ;; can't avoid boxed warning, may be any sort of number

(extend-protocol mp/PSquare
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (square [m]
      (mapmatrix * m m)))

(extend-protocol mp/PRowOperations
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (swap-rows [m i j]
      (let [i (long i)
            j (long j)]
        (if (== i j)
          m
          (assoc (assoc m i (m j)) j (m i)))))
    (multiply-row [m i factor]
      (assoc m i (mp/scale (m i) factor)))
    (add-row [m i j factor]
      (assoc m i (mp/matrix-add (m i) (mp/matrix-multiply (m j) factor)))))


;; helper functin to build generic maths operations
(defn build-maths-function
  ([[name func]]
    `(~name [~'m]
            (mapmatrix (fn [x#] (double (~func (double x#)))) ~'m))))

;; code generation for maths functions
;; we generate both name and name! versions
#?(:clj
(eval
  `(extend-protocol mp/PMathsFunctions
     #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
       ~@(map build-maths-function mops/maths-ops)
       ~@(map (fn [[name func]]
                (let [name (str name "!")
                      mname (symbol name)
                      mpmname (symbol "clojure.core.matrix.protocols" name)]
                  `(~mname [m#]
                     (doseq [s# (mp/get-major-slice-seq m#)]
                       (~mpmname s#)))))
              mops/maths-ops)))
)



;; we need to implement this for all persistent vectors since we need to check all nested components
(extend-protocol mp/PConversion
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (convert-to-nested-vectors [m]
      (if (is-nested-persistent-vectors? m)
        m
        (let [m (mapv-identity-check mp/convert-to-nested-vectors m)
              m-shapes (map mp/get-shape m)]
          (if (every? (partial = (first m-shapes)) (rest m-shapes))
            m
            (error "Can't convert to persistent vector array: inconsistent shape."))))))

(defn- copy-to-double-array! 
  "Copy an arbitrary array to a region of a double array.
   Assumes size represents the element count of the array, must be greater than zero."
  ([m ^doubles arr ^long off ^long size]
    (cond
      ;; handle a single numerical value
      (number? m) (if (== size 1)
                    (aset arr off (double m))
                    (error "Invalid shape while copying to double array"))
      ;; handle a Clojure vector. Could have nested arrays
      (vector? m)
        (let [m ^IPersistentVector m
              ct (count m)
              skip (quot size ct)]
          (dotimes [i ct]
            (let [slc #?(:clj (.nth m i)
                         :cljs (nth m i))]
              (copy-to-double-array! slc arr (+ off (* i skip)) skip))))
      ;; otherwise, must be some arbitrary core.matrix array
      ;; TODO think of a faster way to implement this.
      :else
        (doseq-indexed [v (mp/element-seq m) i]
          (aset arr (+ off i) (double v))))))

(extend-protocol mp/PDoubleArrayOutput
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (to-double-array [m]
      (let [size (long (mp/element-count m))
            arr (double-array size)
            ct (count m)]
        (when (> size 0)
          (copy-to-double-array! m arr 0 size))
        arr))
    (as-double-array [m] nil))

(defn- copy-to-object-array [m ^objects arr ^long off ^long size]
  (let [ct (count m)]
    (cond
      ;; we need this to handle the case of non-vectors nested in vectors
      (not (vector? m))
        (doseq-indexed [v (mp/element-seq m) i]
          (aset arr (+ off i) v))
      ;; m must be a vector from now on
      (and (== size ct) (not (vector? (#?(:clj .nth :cljs nth)
                                       ^#?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer}))) m 0 nil))))
        (dotimes [i size]
          (aset arr (+ off i) (nth ^#?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer}))) m i)))
      :else
        (let [skip (quot size ct)]
          (dotimes [i ct]
            (copy-to-object-array (#?(:clj .nth :cljs nth)
                                   ^#?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer}))) m i) arr (+ off (* i skip)) skip))))
    arr))

(extend-protocol mp/PObjectArrayOutput
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (to-object-array [m]
      (let [size (long (mp/element-count m))
            arr (object-array size)
            ct (count m)]
        (copy-to-object-array m arr 0 size)
        arr))
    (as-object-array [m] nil))

(extend-protocol mp/PFunctionalOperations
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (element-seq [m]
      (cond
        (== 0 (count m))
          nil
        (>= (long (mp/dimensionality (nth m 0))) 1)
          ;; we are a 2D+ array, so be conservative and create a concatenated sequence
          (mapcat mp/element-seq m)
        :else
          ;; we are a 1D vector, so already a valid seqable result for element-seq
          m))
    (element-map
      ([m f]
        (mapmatrix f m))
      ([m f a]
        (let [[m a] (mp/broadcast-same-shape m a)]
          (mapmatrix f m a)))
      ([m f a more]
        (let [arrays (cons m (cons a more))
              shapes (map mp/get-shape arrays)
              sh (or (mp/common-shape shapes) (error "Attempt to do element map with incompatible shapes: " (mapv mp/get-shape arrays)))
              arrays (map #(mp/broadcast % sh) arrays)]
          (apply mapmatrix f arrays))))
    (element-map!
      ([m f]
        (doseq [s m]
          (mp/element-map! s f))
        m)
      ([m f a]
        (dotimes [i (count m)]
          (mp/element-map! (m i) f (mp/get-major-slice a i)))
        m)
      ([m f a more]
        (dotimes [i (count m)]
          (apply mp/element-map! (m i) f (mp/get-major-slice a i) (map #(mp/get-major-slice % i) more)))
        m))
    (element-reduce
      ([m f]
        (reduce f (mp/element-seq m)))
      ([m f init]
        (reduce f init (mp/element-seq m)))))

(extend-protocol mp/PMapIndexed
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (element-map-indexed
      ([ms f]
       (let [dims (long (mp/dimensionality ms))]
         (cond
           (== 0 dims) (f [] (scalar-coerce ms))
           (== 1 dims) (vec (for [i (range (count ms))]
                              (f [i] (nth ms i))))
           :else       (vec (for [i (range (count ms))]
                              (mp/element-map-indexed (nth ms i) #(f (cons i %1) %2)))))))
      ([ms f as]
       (let [as   (mp/broadcast-like ms as)
             dima (long (mp/dimensionality as))]
         (if (mp/is-vector? ms)
           (do
             (when (> dima 1)
               (error "mapping with array of higher dimensionality?"))
             (when (and (== 1 dima)
                        (not= (mp/dimension-count ms 0) (mp/dimension-count as 0)))
               (error "Incompatible vector sizes"))
             (if (== 0 dima)
               (let [v (scalar-coerce as)]
                 (mapv #(f [%1] %2 v) (range (count ms))) ms)
               (mapv #(apply f [%1] %&) (range (count ms)) ms (mp/element-seq as))))
           (mapv (fn [i m a] (mp/element-map-indexed m #(apply f (cons i %1) %&) a))
                 (range (count ms)) ms (mp/get-major-slice-seq as)))))
      ([ms f as more]
       (if (mp/is-vector? ms)
         (apply mapv #(apply f [%1] %&) (range (count ms)) ms as more)
         (apply mapv (fn [i m a & mr]
                       (mp/element-map-indexed m #(apply f (cons i %1) %&) a mr))
                     (range (count ms)) ms as more))))
    (element-map-indexed!
      ([m f]
        (dotimes [i (count m)]
          (mp/element-map-indexed! (m i) #(f (cons i %1) %2)))
        m)
      ([m f a]
        (dotimes [i (count m)]
          (mp/element-map-indexed! (m i) #(apply f (cons i %1) %&)
                                   (mp/get-major-slice a i)))
        m)
      ([m f a more]
        (dotimes [i (count m)]
          (apply mp/element-map-indexed! (m i) #(apply f (cons i %1) %&)
                 (mp/get-major-slice a i) (map #(mp/get-major-slice % i) more)))
        m))
  )

(extend-protocol mp/PSelect
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (select
      ([a args]
       (if (= 1 (count args))
         (do
           (if (= 1 (mp/dimensionality a))
             (apply vector (mapv #(nth a %) (first args)))
             (error "Array dimension does not match length of args")))
         (apply vector (mapv #(mp/select (nth a %) (next args))
                             (first args)))))))

(extend-protocol mp/PIndexImplementation
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
	  (index? [m] true))

;; =====================================
;; Register implementation

(imp/register-implementation [1])


