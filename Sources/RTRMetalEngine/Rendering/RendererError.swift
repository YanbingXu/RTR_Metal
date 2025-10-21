import Foundation

enum RendererError: Error {
    case resourceCreationFailed(String)
    case commandEncodingFailed(String)
    case pipelineCreationFailed(String)
}
