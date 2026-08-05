import Foundation

/// Minimal safetensors reader/writer — enough to consume the goldens and emit a
/// candidate bundle, with no external dependencies.
///
/// Layout: `[u64 header length][JSON header][data blob]`. Every tensor's
/// `data_offsets` are relative to the START OF THE BLOB, not the file. That
/// detail is why renaming keys never invalidates offsets — the same property
/// that let the reference-side encoder remap be a header-only rewrite.
package enum Safetensors {
    package struct TensorInfo: Sendable {
        package let dtype: String
        package let shape: [Int]
        package let begin: Int
        package let end: Int
        package var byteCount: Int { end - begin }
        package var elementCount: Int {
            shape.reduce(1) { partial, dimension in
                partial.multipliedReportingOverflow(by: dimension).partialValue
            }
        }
    }

    package enum Error: Swift.Error, CustomStringConvertible {
        case tooSmall
        case badHeader(String)
        case unsupportedDType(String)
        case missing(String)
        case shapeMismatch(String)
        case notWritable(String)

        package var description: String {
            switch self {
            case .tooSmall: "file too small to be safetensors"
            case .badHeader(let m): "bad safetensors header: \(m)"
            case .unsupportedDType(let d): "unsupported dtype \(d) (goldens are F32)"
            case .missing(let n): "tensor not found: \(n)"
            case .notWritable(let p): "could not create \(p)"
            case .shapeMismatch(let m): "shape mismatch: \(m)"
            }
        }
    }

    /// Header is read with a FileHandle, never by mapping the file: `Data`'s
    /// `.mappedIfSafe` silently declines on very large files and falls back to
    /// a full read, which cost 21s per inspect on the 62 GiB DiT — 66 GB of I/O
    /// to parse a 113 KB header. Tensor data is mapped lazily, on first access.
    package struct Archive {
        package let tensors: [String: TensorInfo]
        package let metadata: [String: String]
        package let url: URL
        private let blobStart: Int
        private let payloadBox: Payload

        /// Holds the payload mapping so it is created **once**, not once per
        /// tensor read.
        ///
        /// This is a class because `Archive` is a struct held by `let`, and the
        /// cost of getting it wrong is not subtle: the previous version
        /// re-evaluated `Data(contentsOf:)` inside every `float32` call, so
        /// reading the 226 taps of a 31 GB production golden re-opened that
        /// file 226 times. Worse, `.mappedIfSafe` *silently declines* on very
        /// large files and falls back to a full read — the same trap already
        /// documented for the 62 GiB checkpoint header — so each of those 226
        /// calls could be a 31 GB read, which is what made a parity run's
        /// comparison phase take ~12 minutes on top of a 20 minute forward.
        ///
        /// `.alwaysMapped` is deliberate: for a multi-GB local file mapping is
        /// the only sane choice, and an outright failure is better than a
        /// silent 31 GB read.
        ///
        /// Not thread-safe. The parity tools are single-threaded; if that ever
        /// changes this needs a lock.
        private final class Payload {
            private let url: URL
            private var data: Data?
            init(url: URL, eager: Data?) { self.url = url; self.data = eager }
            func get() throws -> Data {
                if let data { return data }
                let d = try Data(contentsOf: url, options: .alwaysMapped)
                data = d
                return d
            }
        }

        /// `headerOnly` keeps this to two small reads regardless of file size.
        package init(url: URL, headerOnly: Bool = true) throws {
            self.url = url
            let fh = try FileHandle(forReadingFrom: url)
            defer { try? fh.close() }
            let fileSize = try fh.seekToEnd()
            try fh.seek(toOffset: 0)
            guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
                throw Error.tooSmall
            }
            let n = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
            // Real checkpoint headers are around 100 KB. A hard ceiling stops
            // a hostile eight-byte prefix from asking JSONSerialization for a
            // gigabyte allocation before any tensor has been validated.
            let maximumHeaderBytes: UInt64 = 64 * 1024 * 1024
            guard n > 0, n <= maximumHeaderBytes else {
                throw Error.badHeader("implausible length \(n)")
            }
            guard n <= fileSize - 8 else {
                throw Error.badHeader("header length \(n) exceeds file size \(fileSize)")
            }
            guard let headerData = try fh.read(upToCount: Int(n)), headerData.count == Int(n) else {
                throw Error.tooSmall
            }
            self.payloadBox = Payload(
                url: url,
                eager: headerOnly ? nil : try Data(contentsOf: url, options: .alwaysMapped))
            guard let obj = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
                throw Error.badHeader("not a JSON object")
            }
            var t: [String: TensorInfo] = [:]
            var meta: [String: String] = [:]
            let payloadBytes = Int(fileSize - 8 - n)
            let byteWidth: [String: Int] = [
                "BOOL": 1, "I8": 1, "U8": 1,
                "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                "I32": 4, "U32": 4, "F32": 4,
                "I64": 8, "U64": 8, "F64": 8,
            ]
            for (k, v) in obj {
                if k == "__metadata__" {
                    meta = (v as? [String: String]) ?? [:]
                    continue
                }
                guard let d = v as? [String: Any],
                      let dtype = d["dtype"] as? String,
                      let shape = d["shape"] as? [Int],
                      let off = d["data_offsets"] as? [Int], off.count == 2
                else { throw Error.badHeader("entry \(k)") }
                guard shape.allSatisfy({ $0 >= 0 }) else {
                    throw Error.badHeader("entry \(k) has a negative dimension")
                }
                var elements = 1
                for dimension in shape {
                    let next = elements.multipliedReportingOverflow(by: dimension)
                    guard !next.overflow else {
                        throw Error.badHeader("entry \(k) shape overflows Int")
                    }
                    elements = next.partialValue
                }
                let begin = off[0], end = off[1]
                guard begin >= 0, end >= begin, end <= payloadBytes else {
                    throw Error.badHeader("entry \(k) offsets \(off) outside payload 0...\(payloadBytes)")
                }
                guard let width = byteWidth[dtype] else {
                    throw Error.badHeader("entry \(k) has unsupported safetensors dtype \(dtype)")
                }
                let expected = elements.multipliedReportingOverflow(by: width)
                guard !expected.overflow, end - begin == expected.partialValue else {
                    throw Error.badHeader("entry \(k) byte count does not match \(dtype) shape \(shape)")
                }
                t[k] = TensorInfo(dtype: dtype, shape: shape, begin: off[0], end: off[1])
            }
            let ordered = t.map { (name: $0.key, begin: $0.value.begin, end: $0.value.end) }
                .sorted { ($0.begin, $0.end, $0.name) < ($1.begin, $1.end, $1.name) }
            for pair in zip(ordered, ordered.dropFirst()) where pair.1.begin < pair.0.end {
                throw Error.badHeader("tensor ranges overlap: \(pair.0.name) and \(pair.1.name)")
            }
            self.tensors = t
            self.metadata = meta
            self.blobStart = 8 + Int(n)
        }

        /// Maps the payload on demand, **once** — only readers of tensor data
        /// pay for it, and they pay once rather than per call.
        private func payload() throws -> Data {
            try payloadBox.get()
        }

        package var names: [String] { tensors.keys.sorted() }

        package func info(_ name: String) throws -> TensorInfo {
            guard let i = tensors[name] else { throw Error.missing(name) }
            return i
        }

        /// Goldens are written fp32 precisely so a reader needs no dtype zoo.
        package func float32(_ name: String) throws -> [Float] {
            let i = try info(name)
            guard i.dtype == "F32" else { throw Error.unsupportedDType(i.dtype) }
            let d = try payload()
            let lo = blobStart + i.begin
            let hi = blobStart + i.end
            guard d.count >= hi else { throw Error.tooSmall }
            guard hi > lo else { return [] }
            // One copy, not two. `subdata` allocates and copies the range, and
            // `Array(...)` then copies again — on a 331 MB production block tap
            // that is 662 MB of churn per read, 226 times per run.
            // Offsets are multiples of 4 because every tensor here is F32, so
            // binding at `lo` is aligned.
            return d.withUnsafeBytes { raw in
                let base = raw.baseAddress!.advanced(by: lo)
                    .assumingMemoryBound(to: Float.self)
                return Array(UnsafeBufferPointer(start: base, count: (hi - lo) / 4))
            }
        }

        /// Index tensors — row selections and modulation rows — are the one
        /// non-float thing a fixture carries.
        package func int32(_ name: String) throws -> [Int] {
            let i = try info(name)
            guard i.dtype == "I32" else { throw Error.unsupportedDType(i.dtype) }
            let d = try payload()
            let lo = blobStart + i.begin, hi = blobStart + i.end
            guard d.count >= hi else { throw Error.tooSmall }
            guard hi > lo else { return [] }
            return d.withUnsafeBytes { raw in
                let base = raw.baseAddress!.advanced(by: lo)
                    .assumingMemoryBound(to: Int32.self)
                return UnsafeBufferPointer(start: base, count: (hi - lo) / 4).map(Int.init)
            }
        }
    }

    /// Writes an fp32 bundle in the shape `ingest_candidate.py` expects.
    ///
    /// **Streams.** The obvious implementation — accumulate every tensor into a
    /// `blob`, then append that into an `out`, then write `out` — holds the
    /// whole bundle *three* times: once in `tensors`, once in `blob`, once in
    /// `out`. At production shape the candidate is 33.4 GB, so that is ~100 GB
    /// to write a 33 GB file, on top of a 64.8 GB resident checkpoint. It is
    /// also why `Data.append` is the wrong tool here at all: growing a 33 GB
    /// `Data` reallocates and copies repeatedly, so the transient is worse than
    /// the 3x steady state suggests.
    ///
    /// That cost is invisible for twenty minutes and then arrives all at once
    /// at the end of a run, which reads exactly like a leak and is not one.
    /// It is what killed every `--keep-ada-lnfp32-resident` production run.
    ///
    /// Offsets are known from the shapes alone, so the header can be built
    /// before a single byte is copied and each tensor written straight through.
    /// Extra memory is now one tensor's worth, not the bundle's.
    ///
    /// The byte layout is unchanged — same sorted-key order, same offsets.
    package static func write(
        _ tensors: [String: (shape: [Int], values: [Float])],
        metadata: [String: String] = [:],
        to url: URL
    ) throws {
        let names = tensors.keys.sorted()

        // 1. Header from shapes only. No payload touched.
        var header: [String: Any] = [:]
        var offset = 0
        for name in names {
            let (shape, values) = tensors[name]!
            guard shape.reduce(1, *) == values.count else {
                throw Error.shapeMismatch("\(name): shape \(shape) vs \(values.count) values")
            }
            let begin = offset
            offset += values.count * MemoryLayout<Float>.size
            header[name] = ["dtype": "F32", "shape": shape,
                            "data_offsets": [begin, offset]]
        }
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        let headerData = try JSONSerialization.data(withJSONObject: header,
                                                    options: [.sortedKeys])

        // 2. Create, then stream. Truncating an existing file matters: writing
        //    a shorter bundle over a longer one otherwise leaves a valid header
        //    in front of stale trailing bytes.
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
        guard fm.createFile(atPath: url.path, contents: nil) else {
            throw Error.notWritable(url.path)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }

        var n = UInt64(headerData.count).littleEndian
        try withUnsafeBytes(of: &n) { try handle.write(contentsOf: Data($0)) }
        try handle.write(contentsOf: headerData)

        // 3. Same order the offsets were computed in — a mismatch here would
        //    produce a structurally valid file with every tensor's bytes under
        //    another tensor's name.
        for name in names {
            let values = tensors[name]!.values
            try values.withUnsafeBufferPointer {
                try handle.write(contentsOf: Data(buffer: $0))
            }
        }
    }
}
