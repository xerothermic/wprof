import SwiftUI

/// The six faces, ordered to match the facelet layout below.
enum CubeFace: Int, CaseIterable {
    case up = 0, down, front, back, left, right

    var stickerColor: Color {
        switch self {
        case .up: .white
        case .down: .yellow
        case .front: .green
        case .back: .blue
        case .left: .orange
        case .right: .red
        }
    }
}

/// State of a 3x3 cube as 54 facelets.
///
/// Faces are ordered U, D, F, B, L, R; each face holds 9 facelets in row-major
/// order (`row * 3 + col`), viewed head-on with the standard net orientation
/// (U's top row toward the back, F's top row up, etc.). A facelet's value is the
/// `CubeFace.rawValue` of the colour currently on it.
struct CubeState {
    private(set) var facelets: [Int]

    init() {
        facelets = (0..<54).map { $0 / 9 }
    }

    /// Source-form permutation for one clockwise quarter turn of each face:
    /// after the turn, `new[i] = old[perm[i]]`. These tables were generated from
    /// a geometric cubie model and verified against standard identities
    /// (X⁴ = I, X·X′ = I, (R U R′ U′)⁶ = I).
    private static let perms: [Character: [Int]] = [
        "U": [6, 3, 0, 7, 4, 1, 8, 5, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 45, 46, 47, 21, 22, 23, 24, 25, 26, 36, 37, 38, 30, 31, 32, 33, 34, 35, 18, 19, 20, 39, 40, 41, 42, 43, 44, 27, 28, 29, 48, 49, 50, 51, 52, 53],
        "D": [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 12, 9, 16, 13, 10, 17, 14, 11, 18, 19, 20, 21, 22, 23, 42, 43, 44, 27, 28, 29, 30, 31, 32, 51, 52, 53, 36, 37, 38, 39, 40, 41, 33, 34, 35, 45, 46, 47, 48, 49, 50, 24, 25, 26],
        "F": [0, 1, 2, 3, 4, 5, 44, 41, 38, 51, 48, 45, 12, 13, 14, 15, 16, 17, 24, 21, 18, 25, 22, 19, 26, 23, 20, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 9, 39, 40, 10, 42, 43, 11, 6, 46, 47, 7, 49, 50, 8, 52, 53],
        "B": [47, 50, 53, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 36, 39, 42, 18, 19, 20, 21, 22, 23, 24, 25, 26, 33, 30, 27, 34, 31, 28, 35, 32, 29, 2, 37, 38, 1, 40, 41, 0, 43, 44, 45, 46, 17, 48, 49, 16, 51, 52, 15],
        "L": [35, 1, 2, 32, 4, 5, 29, 7, 8, 18, 10, 11, 21, 13, 14, 24, 16, 17, 0, 19, 20, 3, 22, 23, 6, 25, 26, 27, 28, 15, 30, 31, 12, 33, 34, 9, 42, 39, 36, 43, 40, 37, 44, 41, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53],
        "R": [0, 1, 20, 3, 4, 23, 6, 7, 26, 9, 10, 33, 12, 13, 30, 15, 16, 27, 18, 19, 11, 21, 22, 14, 24, 25, 17, 8, 28, 29, 5, 31, 32, 2, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47],
    ]

    private mutating func turn(_ face: Character) {
        let perm = Self.perms[face]!
        facelets = perm.map { facelets[$0] }
    }

    mutating func apply(move: String) {
        guard let face = move.first else { return }
        let times: Int
        switch move.dropFirst() {
        case "'": times = 3
        case "2": times = 2
        default: times = 1
        }
        for _ in 0..<times { turn(face) }
    }

    static func scrambled(by moves: [String]) -> CubeState {
        var cube = CubeState()
        for move in moves { cube.apply(move: move) }
        return cube
    }

    func color(face: CubeFace, row: Int, col: Int) -> Color {
        CubeFace(rawValue: facelets[face.rawValue * 9 + row * 3 + col])!.stickerColor
    }
}
