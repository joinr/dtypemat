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
            [tech.v3.datatype.functional :as fun]
            [tech.v3.datatype.argops :as argops]
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

(extend-protocol mp/PDimensionInfo
;;  #?(:clj IPersistentVector :cljs PersistentVector)
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (dimensionality [m]  (count (dtype/shape m))) ;;TOM check...
  (is-vector? [m]      (== (count (dtype/shape m)) 1))
  (is-scalar? [m] false)
  (get-shape [m] (dtype/shape m))
  (dimension-count [m x] (nth (dtype/shape m) x)))

(extend-protocol mp/PIndexedAccess
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (get-1d [m x]    (dtt/mget m x))
  (get-2d [m x y] (dtt/mget m x y))
  (get-nd [m indexes] (apply dtt/mget m indexes)))

;; we extend this so that nested mutable implementions are possible
(extend-protocol mp/PIndexedSetting
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (set-1d [m row v]        (-> m  dtt/clone (dtt/mset!  row v)))
  (set-2d [m row column v] (-> m  dtt/clone (dtt/mset!   row column v)))
  (set-nd [m indexes v]    (apply dtt/mset! (dtt/clone m) indexes v))
  (is-mutable? [m]         true))

(extend-protocol mp/PIndexedSettingMutable
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (set-1d! [m row v]        (dtt/mset! m row v))
  (set-2d! [m row column v] (dtt/mset! m row column v))
  (set-nd! [m indexes v]    (apply dtt/mset! m indexes v)))

(extend-protocol mp/PMatrixCloning
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (clone [m] (dtt/clone m)))

(extend-protocol mp/PTypeInfo
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (element-type [m]
    (case (dtype/elemwise-datatype m)
      :int8  Byte
      :int16 Short
      :char  Character
      :int32 Integer
      :int64 Long

      ;;unsigned though...
      ;;Maybe emit warning for unsigned types..
      :uint8  Byte
      :uint16 Short
      :uint32 Integer
      :uint64 Long

      :float32 Float
      :float64 Double
      :boolean Boolean
      :object  Object
      (throw (ex-info "unknown primitive type!" {:in m :type (dtype/elemwise-datatype m)})))))

(extend-protocol mp/PElementCount
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (element-count [m] (dtype/ecount m)))

(extend-protocol mp/PArrayMetrics
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (nonzero-count [m] ;;TOM: lame, we don't cache these explicitly....
    (count (argops/argfilter #(not (zero? %)) m))))

(extend-protocol mp/PValidateShape
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (validate-shape
    ([m]
     ;;operation doesn't make sense to me for a single ndbuffer....
     ;;for a singleton it's always true.  this seems dumb.
     true)
    ([m shape]
     (let [s (dtype/shape m)]
       (if (= s shape)
         s
         (throw (ex-info (str "Inconsistent shape for persistent vector array, expected: " shape " on array " m)
                         {:m m :shape shape})))))))

(extend-protocol mp/PRowColMatrix
  #_"Protocol to support construction of row and column matrices from 1D vectors.
   A vector of length N should be converted to a 1xN or Nx1 matrix respectively.
   Should throw an error if the data is not a 1D vector"
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (column-matrix [m data]
    (-> (dtt/->tensor [data] :datatype (dtype/get-datatype m))
        (dtt/transpose [1 0])))
  (row-matrix [m data]
    (dtt/->tensor [data] :datatype (dtype/get-datatype m))))

(extend-protocol mp/PMutableMatrixConstruction
  #_"Protocol for creating a mutable copy of a matrix. If implemented, must return either a fully mutable
   copy of the given matrix, or nil if not possible.
   The default implementation will attempt to choose a suitable mutable matrix implementation."
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (mutable-matrix [m] (dtt/clone m)))

(extend-protocol mp/PMutableCoercion
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  #_"Protocol for coercing to a mutable format. May return the original array, if it is already fully mutable,
   otherwise should return a fully mutable copy of the array.
   Should return nil to indicate that this implementation cannot create a mutable array from the given data.
   The default implementation will attempt to choose a suitable mutable matrix implementation."
  (ensure-mutable [m] m))


(extend-protocol mp/PSparse
  #_
  "Protocol for constructing a sparse array from the given data. Implementations should
   consider the possibility that data may be a large lazy sequence, possibly larger than memory, so should ideally
   attempt to construct the sparse matrix incrementally without realising the whole sequence at once.
   May return nil if no sparse conversion is available."
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  ;;TOM: for now we just assume dense tensors.  I think there's a sparse encoding though.
  (sparse-coerce [m data] nil)
  (sparse        [m] nil))

(extend-protocol mp/PNative
  #_"Protocol for creating and handling native arrays. Implementations must return a native format array if they
   are able to, or nil otherwise."
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (native [m]  (dtt/ensure-native m))
  (native? [m] (some? (dtype/as-native-buffer m))))

(extend-protocol mp/PDense
  #_"Protocol for constructing a dense array from the given data."
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (dense-coerce [m data] (dtt/->tensor data :datatype (dtype/get-datatype m)))
  (dense [m] m))

(extend-protocol mp/PImmutableMatrixConstruction
  #_"Protocol for creating an immutable copy of a matrix. If implemented, must return a fully immutable
   copy of the given matrix.
   The default implementation will attempt to choose a suitable immutable matrix implementation."
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  ;;TOM: for now, we have cloned ops on tensors.  May need a type wrapper, dunno!
  (immutable-matrix [m] (dtt/clone m)))

#_
(extend-protocol mp/PZeroDimensionConstruction
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (new-scalar-array
    ([m] 0)
    ([m value])
    "Construct a new zero-dimensional array with the specified scalar value (zero if not specified)"))

#_
(extend-protocol mp/PZeroDimensionAccess
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  #_"Protocol for accessing the scalar value in zero-dimensional arrays. Zero dimensional arrays differ
   from scalar values in the following two senses:
    - They may be mutable (in which case set-0d! is expected to work)
    - They are not considered themselves to be scalars. Hence you must use get-0d to access the
      contained scalar value"
  (get-0d
    [m]
    "Gets the scalar value in an 0d array.")
  (set-0d!
    [m value]
    "Sets the scalar value in the 0d array to a given value. Throws an error if not mutable."))

#_
(defprotocol PZeroDimensionSet
  "Protocol for setting the scalar value in zero-dimensional arrays."
  (set-0d [m value] "Sets the scalar value in a 0-d array, returning a new 0-d array"))


;;TOM: maybe do these with computed tensors?
#_
(defprotocol PSpecialisedConstructors
  "Protocol for construction of special matrices."
  (identity-matrix
    [m dims]
    "Create a 2D identity matrix with the given number of dimensions")
  (diagonal-matrix
    [m diagonal-values]
    "Create a diagonal matrix with the specified leading diagonal values"))

#_
(defprotocol PPermutationMatrix
  "Protocol for construction of a permutation matrix."
  (permutation-matrix [m permutation]))

#_
(defprotocol PBlockDiagonalMatrix
  "Protocol for construction of a block diagonal matrix."
  (block-diagonal-matrix [m blocks]))

(extend-protocol mp/PCoercion
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (coerce-param [m param]
    (if (dtt/tensor? param)
      param
      (let [param (if (seq param) ;;TOM uncertain
                    param
                    [param])] ;;Do we interpret scalars as 0d tensors?  Probably an ERROR
        (dtt/->tensor param :datatype (dtype/get-datatype m))))))

(extend-protocol mp/PBroadcast
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (broadcast [m target-shape]
    (dtt/broadcast m target-shape)))

(extend-protocol mp/PBroadcastLike
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (broadcast-like [m a]
      (mp/broadcast a (mp/get-shape m))))

;;waiting....
(extend-protocol mp/PBroadcastCoerce
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (broadcast-coerce [m a]
    (if  (dtt/tensor? a)
      (if (mp/same-shape? m a)
        a
        (dtt/broadcast a   (dtype/shape m)))
      (let [donor-type   (dtype/get-datatype m)]
        (dtt/broadcast (dtt/->tensor a :datatype donor-type) (dtype/shape m))))))

;; we need to implement this for all persistent vectors since we need to check all nested components
(extend-protocol mp/PConversion
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (convert-to-nested-vectors [m]
    (mapv vec m) ;;TOM 2x check
    #_
    (if (is-nested-persistent-vectors? m)
      m
      (let [m (mapv-identity-check mp/convert-to-nested-vectors m)
            m-shapes (map mp/get-shape m)]
        (if (every? (partial = (first m-shapes)) (rest m-shapes))
          m
          (error "Can't convert to persistent vector array: inconsistent shape."))))))

(extend-protocol mp/PReshaping
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  #_
  "Protocol to reshape matrices. Should support any new shape allowed by the implementation.
   Must preserve row-major ordering of matrix elements.
   If the original matrix is mutable, must return a new mutable copy of data.
   If the new shape has less elements than the original shape, it is OK to truncate the remaining elements.
   If the new shape requires more elements than the original shape, should throw an exception."
  (reshape [m shape]
    (dtt/reshape (dtt/clone m) shape)))

(extend-protocol mp/PReshapeView
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  #_
  "Protocol to reshape matrices. Guarantees a view over the original data if mutable.
   If the new shape has less elements than the original shape, must truncate the remaining elements.
   Behaviour is undefined if the new shape requires more elements than the original shape."
  (reshape-view [m shape] (dtt/reshape m shape)))

(extend-protocol mp/PPack
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
#_
  "Protocol to efficiently pack an array, according to the most efficient representation for a given
   implementation.
   Definition of pack is up to the implementation to interpret, but the general rules are:
   1. Must not change the value of the array for comparison purposes
   2. Must not change the shape of the array
   3. May preserve sparse representation
   4. Should convert to most efficient format for common operations (e.g. mget, inner-product)"

  ;;TOM: unsure!
  (pack [m] (tech.v3.datatype.packing/pack m)))

;;TOM: revisit, is there an optimized rep?
(extend-protocol mp/PSameShape
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  #_
  "Protocol to test if two arrays have the same shape. Implementations may have an optimised
   method for shape equality tests, and this is a frequently required operations so it may
   make sense to provide an optimised implementation."
  (same-shape? [a b]
    (= (dtype/shape a) (dtype/shape b))))

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

(extend-protocol mp/PSliceSeq2
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  ;;TOM: I am guessing based on the docs that this is right...but 2X Check!
  (get-slice-seq [m dim]
    (dtt/slice m dim)))

(extend-protocol mp/PSliceViewSeq
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  #_
  "Returns the row-major slice views of the array.
   These must be arrays if the array is mutable, i.e. slices of a 1D vector
   must be 0-dimensional mutable arrays."
  ;;TOM: 2x check; there's a vernacular difference between view and slice, so
  ;;we may need immutable ops.  I think we're fine for now...
  (get-major-slice-view-seq [m] (dtt/rows m)))

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

;;TOM: TBD need to think this through.  Unsure about the semantics!
#_
(extend-protocol mp/PSliceJoinAlong
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  #_
  "Protocol for concatenating / joining arrays."
  (join-along [m a dim] "Concatenates a to m, along the slice dimension dim"))

(extend-protocol mp/PSubVector
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (subvector [m start length]
    (dtt/select  m (range start (+ (long start) (long length))))))

;;TOM: todo, probably a compute matrix
#_
(defprotocol PMatrixSubComponents
  "Protocol for picking out subsections of a 2D matrix. Should return a mutable view if possible.
   The default implementation creates a new vector containing the diagonal values."
  (main-diagonal [m]
    "Returns the main (leading) diagonal of a matrix."))

(extend-protocol mp/PSparseArray
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
     #_
  "Protocol for determining if an array is in a sparse format. It is up to the implementation to define
   its own sparse formats, but in general the intention should be that a sparse array uses significantly
   less storage than an equivalent dense array, assuming a high proportion of zero values in the array."
  (is-sparse? [m] false))

(extend-protocol mp/PNewSparseArray
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  #_
  "Protocol for constructing sparse arrays. Should return nil if the sparse array shape is not supported."
  (new-sparse-array [m shape] nil ))

(extend-protocol mp/PZeroCount
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (zero-count [m] ;;TOM: lame, we don't cache these explicitly....
    (count (argops/argfilter #(zero? %) m))))


;;PENDING
#_
(extend-protocol mp/PAssignment
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  #_"Protocol for assigning values element-wise to mutable arrays."
  (assign! [m source]
    
    #_"Sets all the values in an array from a given source. Source may be a scalar
     or any smaller array that can be broadcast to the shape of m.")
  (assign-array!
    ([m arr])
    ([m arr start length])
    #_"Sets the elements in an array from a Java array source, in row-major order."))


(extend-protocol mp/PRotate
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (rotate [m dim places]
    (let [dims (vec (repeat (mp/dimensionality m) 0))] ;;TOM maybe inefficient.
      (dtt/rotate m (assoc dims dim places)))))

(extend-protocol mp/PTransposeDims
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (transpose-dims [m ordering]
      (dtt/transpose m ordering)))

(extend-protocol mp/POrder
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (order
    ([m indices]  (dtt/select m indices))
    ([m dimension indices]
     (if (zero? dimension)
       (dtt/select m indices)
       (let [dims (vec (repeat (mp/dimensionality m) :all))]
         ;;TOM maybe better way but meh.
         (apply dtt/select m (assoc dims dimension indices)))))))

(extend-protocol mp/PMatrixAdd
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (matrix-add [m a]
      (let [a (mp/broadcast-coerce m a)]
        (fun/+ m a)))
    (matrix-sub [m a]
      (let [a (mp/broadcast-coerce m a)]
        (fun/-  m a))))

(extend-protocol mp/PVectorOps
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (vector-dot [a b]  (fun/* a b)) ;;TOM verify semantics....
    (length [a] (fun/magnitude a))
    (length-squared [a] (fun/magnitude-squared a))
    (normalise [a]      (fun/* a (/ 1.0 (fun/magnitude a)))))

(extend-protocol mp/PMutableMatrixConstruction
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (mutable-matrix [m]  m)) ;;TOM uncertain

(extend-protocol mp/PImmutableMatrixConstruction
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (immutable-matrix [m] ;;TOM uncertain...
    nil))

(extend-protocol mp/PVectorDistance
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (distance [a b]
    (if (dtt/tensor? b)
      (fun/distance a b)
      (mp/length (mp/matrix-sub b a)))))

(extend-protocol mp/PSummable
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (element-sum [a]
      (fun/reduce-+ a)))



;;TOM 2x check, uncertain if this holds for arbitrary dimensions...
(defn tensor= [l r]
  (and (= (dtype/shape l) (dtype/shape r))
       (->> (map (fn [lrow rrow]
                   (let [j (count lrow)]
                     (reduce (fn [acc j]
                               (if (== (lrow j) (rrow j))
                                 true
                                 (reduced false))) true (range j))))
                 (dtt/rows l) (dtt/rows r))
            (every? identity))))

;;TOM 2x check....unsure if this works correctly.  Appears to.
(defn row= [l r]
  (and (= (mp/get-shape l) (mp/get-shape r))
       (->> (map (fn [lrow rrow]
                   (let [j (count lrow)]
                     (reduce (fn [acc j]
                               (if (== (mp/get-1d lrow j) (mp/get-1d rrow j))
                                 true
                                 (reduced false))) true (range j))))
                 (mp/get-rows l) (mp/get-rows r))
            (every? identity))))

(extend-protocol mp/PMatrixEquality
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (matrix-equals [a b]
      (let [bd (mp/dimensionality a)
            ad (mp/dimensionality b)]
        (and (= bd ad)
             (cond (zero? bd)
                     (== bd ad)e
                   (dtt/tensor? b)
                     (tensor= a b)
                   :else
                     (row=   a b))))))

(extend-protocol mp/PMatrixMultiply
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (element-multiply [m a]
      (if (number? a)
        (fun/* m a)
        (let [[m a] (mp/broadcast-compatible m a)]
          (mp/element-map m * a))))
    ;;TOM 2x check, this is probably reallllly slow.
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

;;TOM 2x check....probably REALLY slow.
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

;;TOM 2x check, no idea..
(extend-protocol mp/PMatrixScaling
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (scale [m a]
    (fun/* m a)
      #_(mapmatrix #(* % a) m)) ;; can't avoid boxed warning, may be any sort of number
  (pre-scale [m a]
    (fun/* m a)
      #_(mapmatrix #(* a %) m))) ;; can't avoid boxed warning, may be any sort of number

(extend-protocol mp/PSquare
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (square [m]
    (fun/sq m)
      #_(mapmatrix * m m)))

(extend-protocol mp/PRowOperations
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
    (swap-rows [m i j]
      (let [i (long i)
            j (long j)]
        (if (== i j)
          m
          (dtt/select m [j i])
          #_
          (assoc (assoc m i (m j)) j (m i)))))
    (multiply-row [m i factor]
      ;;TOM 2x check, mutable semantics.
      (dtt/mset! t 0 (fun/* (t 0) factor)))
    ;;TOM 2x check, mutable semantics.
    (add-row [m i j factor]
      (dtt/mset! t i
                 (fun/+ (t i)
                        (fun/* (m j) factor)))
      #_
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

(extend-protocol mp/PMathsFunctions
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (abs [m] (fun/abs m))
  (acos [m] (fun/cos m))
  (asin [m] (fun/asin m))
  (atan [m] (fun/atan m))
  (cbrt [m] (fun/cbrt m))
  (ceil [m] (fun/ceil m))
  (cos [m] (fun/cos m))
  (cosh [m] (fun/cosh m))
  (exp [m] (fun/exp m))
  (floor [m] (fun/floor m))
  (log [m] (fun/log m))
  (log10 [m] (fun/log10 m))
  (round [m] (fun/rint m))
  (signum [m] (fun/signum m))
  (sin [m] (fun/sin m))
  (sinh [m] (fun/sinh m))
  (sqrt [m] (fun/sqrt m))
  (tan [m] (fun/tan m))
  (tanh [m] (fun/tanh m))
  (to-degrees [m] (fun/to-degrees m))
  (to-radians [m] (fun/to-radians m)))

(extend-protocol mp/PMathsFunctionsMutable
  #?(:clj NDBuffer :cljs (throw (ex-info "no cljs support yet!" {:type :dtype.next/ndbuffer})))
  (abs! [m] (fun/abs m))
  (acos! [m] (fun/cos m))
  (asin! [m] (fun/asin m))
  (atan! [m] (fun/atan m))
  (cbrt! [m] (fun/cbrt m))
  (ceil! [m] (fun/ceil m))
  (cos! [m] (fun/cos m))
  (cosh! [m] (fun/cosh m))
  (exp! [m] (fun/exp m))
  (floor! [m] (fun/floor m))
  (log! [m] (fun/log m))
  (log10! [m] (fun/log10 m))
  (round! [m] (fun/rint m))
  (signum! [m] (fun/signum m))
  (sin! [m] (fun/sin m))
  (sinh! [m] (fun/sinh m))
  (sqrt! [m] (fun/sqrt m))
  (tan! [m] (fun/tan m))
  (tanh! [m] (fun/tanh m))
  (to-degrees! [m] (fun/to-degrees m))
  (to-radians! [m] (fun/to-radians m)))


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
  (element-seq [m] (dtype/as-buffer m))
  (element-map
    ([m f]
     (dtype/emap f (dtype/get-datatype m) m))
    ([m f a]
     (if (dtt/tensor? a)
       (dtype/emap f (dtype/get-datatype m) m a)
       ;;TOM 2x check.  do we want to cop this generic path?
       (let [[m a] (mp/broadcast-same-shape m a)] ;;generic but weird.
         (mapmatrix f m a))))
      ;;TOM 2x check, can we keep this all in tensorland?  Should be able to.
      ([m f a more]
        (let [arrays (cons m (cons a more))
              shapes (map mp/get-shape arrays)
              sh (or (mp/common-shape shapes) (error "Attempt to do element map with incompatible shapes: " (mapv mp/get-shape arrays)))
              arrays (map #(mp/broadcast % sh) arrays)]
          (apply mapmatrix f arrays))))
  ;;TOM 2x check if we are using dense semantics...
    (element-map!
      ([m f]
       (let [w1    (dtype/->writer m)
             bound (count w1)]
         (loop [idx 0]
           (if (< idx bound)
             (do (dtype/set-value! w1 idx (f (w1 idx)))
                 (recur (unchecked-inc idx)))
             m))))
      ([m f a]
       (let [w1    (dtype/->writer m)
             a     (dtype/->reader (mp/broadcast-coerce m a))
             bound (count w1)]
         (loop [idx 0]
           (if (< idx bound)
             (do (dtype/set-value! w1 idx (f (w1 idx) (a idx)))
                 (recur (unchecked-inc idx)))
             m))))
      ([m f a more]
       (let [w1        (dtype/->writer m)
             bound     (count w1)
             arrays    (cons m (cons a more))
             shapes (map mp/get-shape arrays)
             sh     (or (mp/common-shape shapes)
                        (error "Attempt to do element map with incompatible shapes: "
                               (mapv mp/get-shape arrays)))
             readers (into [w1]
                           (map (fn [a] (dtype/->reader (mp/broadcast-coerce m a))))
                           (rest arrays))
             row-bound  (count readers)
             nth-entry (apply juxt readers)]
         (loop [idx 0]
           (if (< idx bound)
             (do
               (dtype/set-value! w1 idx (apply f (nth-entry idx)))
               (recur (unchecked-inc idx)))
             m)))))
    ;;TOM 2x check, these are extremely slow paths.  I think we can get
    ;;much much better if we use typed reductions via e.g. tech.v3.datatype.reductions or
    ;;the like.
    (element-reduce
      ([m f]
       (reduce f (dtype/as-buffer m)))
      ([m f init]
       (reduce f init (dtype/as-buffer m)))))

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


