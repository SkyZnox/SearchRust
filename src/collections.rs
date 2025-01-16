use std::collections::HashMap;
use rand::Rng;
use uuid::{Uuid};
use crate::{EMBEDDING_SIZE, TOP_N_RESULTS};

/// Database
/// collections: Hash Map of String and Vec of u8
///
/// collections used to stock all the collections of all the documents
/// The string is the name of the collection
pub struct Database {
    collections: HashMap<String, Collection>,
}

/// Collection
/// documents: Hash Map of Uuid and Vec of u8
/// name: String
///
/// documents used to stock the embeddings of each document
/// name used to recognise the collection
#[derive(Clone)]
pub struct Collection {
    documents: HashMap<Uuid, Vec<u8>>,
    name: String,
}

/// Results
/// results: Vec of tuple of Uuid and f32
///
/// results used to stock the id of the document and the similarity
pub struct Results{
    pub(crate) results: Vec<(Uuid, f32)>,
}


impl Collection {

    /// Constructor
    pub fn new() -> Self {
        Collection {
            documents: HashMap::new(),
            name: String::new(),
        }
    }

    /// Add embedding
    /// Used to add a document to a collection with embedding
    pub fn add_embedding(&mut self, id: Uuid) {
        let embedding = Self::generate_numeric_embedding();
        self.documents.insert(id.clone(), embedding);
    }

    /// Generate numeric embedding
    /// Used to generate fake embeddings
    fn generate_numeric_embedding() -> Vec<u8> {
        let mut rng = rand::thread_rng();
        (0..EMBEDDING_SIZE).map(|_| rng.gen_range(0..9)).collect()
    }

    /// Cosine similarity
    /// Used to calculate the similarity between the query embedding and all other embeddings
    fn cosine_similarity(vec1: &[u8], vec2: &[u8]) -> f32 {
        let dot_product: f32 = vec1.iter().zip(vec2.iter())
            .map(|(a, b)| (*a as f32) * (*b as f32))
            .sum();

        let norm1: f32 = vec1.iter()
            .map(|x| (*x as f32) * (*x as f32))
            .sum::<f32>()
            .sqrt();

        let norm2: f32 = vec2.iter()
            .map(|x| (*x as f32) * (*x as f32))
            .sum::<f32>()
            .sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Search
    /// Return results of the similarity of the n higher rates
    pub fn search(&self, query: &str) -> Results {
        let query_embedding = Self::generate_query_embedding(query);

        let mut results = Vec::new();

        for (doc_id, doc_embedding) in &self.documents {
            let similarity = Self::cosine_similarity(&query_embedding, doc_embedding);

            results.push((doc_id.clone(), similarity));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results.truncate(TOP_N_RESULTS);

        Results { results }
    }

    /// Generate query embedding
    /// Return a fake embedding of the query (non-used actually)
    fn generate_query_embedding(query: &str) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        (0..EMBEDDING_SIZE).map(|_| rng.gen_range(0..9)).collect()
    }

    /// Get length
    /// Return an int, the length of the collection
    pub fn get_length(&self) -> usize {
        self.documents.len()
    }


}


