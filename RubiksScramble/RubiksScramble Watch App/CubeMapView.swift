import SwiftUI

/// 2D unfolded "net" of the cube after the scramble is applied:
///
///         U
///     L   F   R   B
///         D
struct CubeMapView: View {
    let moves: [String]

    private let cell: CGFloat = 10
    private let stickerGap: CGFloat = 1
    private let faceGap: CGFloat = 3

    private var faceSide: CGFloat { cell * 3 + stickerGap * 2 }

    var body: some View {
        let cube = CubeState.scrambled(by: moves)
        NavigationStack {
            ScrollView([.horizontal, .vertical]) {
                VStack(spacing: faceGap) {
                    row(cube, [nil, .up, nil, nil])
                    row(cube, [.left, .front, .right, .back])
                    row(cube, [nil, .down, nil, nil])
                }
                .padding(6)
                .frame(maxWidth: .infinity)
            }
            .navigationTitle("Cube")
        }
    }

    private func row(_ cube: CubeState, _ faces: [CubeFace?]) -> some View {
        HStack(spacing: faceGap) {
            ForEach(Array(faces.enumerated()), id: \.offset) { _, face in
                if let face {
                    faceGrid(cube, face)
                } else {
                    Color.clear.frame(width: faceSide, height: faceSide)
                }
            }
        }
    }

    private func faceGrid(_ cube: CubeState, _ face: CubeFace) -> some View {
        VStack(spacing: stickerGap) {
            ForEach(0..<3, id: \.self) { r in
                HStack(spacing: stickerGap) {
                    ForEach(0..<3, id: \.self) { c in
                        RoundedRectangle(cornerRadius: 1.5)
                            .fill(cube.color(face: face, row: r, col: c))
                            .frame(width: cell, height: cell)
                    }
                }
            }
        }
    }
}

#Preview {
    CubeMapView(moves: ScrambleGenerator.generate())
}
