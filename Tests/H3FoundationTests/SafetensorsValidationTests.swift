// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import H3Foundation

@Suite("safetensors hostile input")
struct SafetensorsValidationTests {
    @Test("a valid bounded tensor is accepted")
    func valid() throws {
        let url = try fixture([
            "x": ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]],
        ], payload: Data(repeating: 0, count: 4))
        defer { try? FileManager.default.removeItem(at: url) }
        let archive = try Safetensors.Archive(url: url)
        #expect(archive.tensors["x"]?.byteCount == 4)
    }

    @Test("an offset outside the payload is refused")
    func outOfBounds() throws {
        let url = try fixture([
            "x": ["dtype": "F32", "shape": [2], "data_offsets": [0, 8]],
        ], payload: Data(repeating: 0, count: 4))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: Safetensors.Error.self) { try Safetensors.Archive(url: url) }
    }

    @Test("shape multiplication overflow is refused")
    func shapeOverflow() throws {
        let url = try fixture([
            "x": ["dtype": "F32", "shape": [Int.max, 2], "data_offsets": [0, 0]],
        ], payload: Data())
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: Safetensors.Error.self) { try Safetensors.Archive(url: url) }
    }

    @Test("overlapping tensor ranges are refused")
    func overlap() throws {
        let url = try fixture([
            "a": ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]],
            "b": ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]],
        ], payload: Data(repeating: 0, count: 4))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: Safetensors.Error.self) { try Safetensors.Archive(url: url) }
    }

    private func fixture(_ header: [String: Any], payload: Data) throws -> URL {
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(headerData.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(headerData)
        data.append(payload)
        let url = FileManager.default.temporaryDirectory
            .appending(path: "h3-safetensors-\(UUID().uuidString)")
        try data.write(to: url, options: .atomic)
        return url
    }
}
