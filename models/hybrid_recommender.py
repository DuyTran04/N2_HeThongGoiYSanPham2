import pandas as pd
import torch
import numpy as np
import pickle
from models.deep_matrix_factorization import DeepMatrixFactorization

class HybridRecommender:
    def __init__(self, data=None, keywords_data=None, dmf_model_path='data/dmf_model.pkl', 
                 popularity_model_path='data/popularity_model.pkl', dmf_weight=0.7, pop_weight=0.3):
        """
        Initialize hybrid recommender
        
        Args:
            data: DataFrame containing product data (optional if models are loaded from files)
            keywords_data: DataFrame containing keywords data for personalization
            dmf_model_path: Path to the DMF model pickle file
            popularity_model_path: Path to the popularity model pickle file
            dmf_weight: Weight for Deep Matrix Factorization recommendations
            pop_weight: Weight for Popularity-based recommendations
        """
        self.data = data
        self.keywords_data = keywords_data
        self.dmf_weight = dmf_weight
        self.pop_weight = pop_weight
        
        # Load DMF model
        self.load_dmf_model(dmf_model_path)
        
        # Load Popularity model
        self.load_popularity_model(popularity_model_path)
    
    def load_dmf_model(self, model_path):
        """Load Deep Matrix Factorization model from file"""
        try:
            with open(model_path, 'rb') as f:
                dmf_info = pickle.load(f)
                self.dmf_model = DeepMatrixFactorization(dmf_info['n_users'], dmf_info['n_items'])
                self.dmf_model.load_state_dict(dmf_info['model_state_dict'])
                self.product_id_encoder = dmf_info['product_encoder']
                self.user_id_encoder = dmf_info.get('user_encoder')
                print("DMF model loaded successfully!")
        except Exception as e:
            print(f"Error loading DMF model: {e}")
            self.dmf_model = None
            self.product_id_encoder = None
            self.user_id_encoder = None
    
    def load_popularity_model(self, model_path):
        """Load Popularity model from file"""
        try:
            with open(model_path, 'rb') as f:
                popularity_info = pickle.load(f)
                self.popularity_model = popularity_info['model']
                self.popularity_formula = popularity_info['score_formula']
                print("Popularity model loaded successfully!")
        except Exception as e:
            print(f"Error loading Popularity model: {e}")
            # Create popularity model from data if available
            if self.data is not None:
                self._create_popularity_model_from_data()
            else:
                self.popularity_model = None
    
    def _create_popularity_model_from_data(self):
        """Create popularity model from data if no precomputed model is available"""
        if self.data is None:
            print("No data available to create popularity model")
            self.popularity_model = None
            return
            
        # Convert columns to numeric
        self.data['rating'] = pd.to_numeric(self.data['rating'], errors='coerce')
        self.data['rating_count'] = pd.to_numeric(self.data['rating_count'], errors='coerce')
        
        # Create purchase_count_estimated if not exists
        if 'purchase_count_estimated' not in self.data.columns:
            self.data['purchase_count_estimated'] = self.data['rating_count'] * 0.5
        
        # Fill NA values
        self.data['rating'] = self.data['rating'].fillna(0)
        self.data['rating_count'] = self.data['rating_count'].fillna(0)
        self.data['purchase_count_estimated'] = self.data['purchase_count_estimated'].fillna(0)
        
        # Calculate popularity score
        self.data['popularity_score'] = (
            (self.data['rating'] * 25) +
            (self.data['rating_count'] * 1.5) +
            (self.data['purchase_count_estimated'] * 0.5)
        )
        
        # Sort by popularity score
        self.popularity_model = self.data.sort_values('popularity_score', ascending=False)
        self.popularity_formula = {
            'rating_weight': 25,
            'rating_count_weight': 1.5,
            'purchase_count_weight': 0.5
        }
        print("Popularity model created from data")
    
    def get_dmf_recommendations(self, product_id, n_recommendations=5):
        """Get recommendations using Deep Matrix Factorization"""
        # Only proceed if model and encoder exist
        if self.dmf_model is None or self.product_id_encoder is None:
            print("DMF model or encoder not loaded")
            return []
            
        try:
            # Transform product_id to encoded id
            encoded_product_id = self.product_id_encoder.transform([product_id])[0]
            
            self.dmf_model.eval()
            with torch.no_grad():
                # Get all product embeddings
                all_products = torch.arange(len(self.product_id_encoder.classes_))
                target_product = torch.tensor([encoded_product_id])
                
                # Get predictions
                similarities = self.dmf_model(
                    target_product.repeat(len(all_products)), 
                    all_products
                ).numpy()
                
                # Get top recommendations
                top_indices = np.argsort(similarities)[-n_recommendations-1:][::-1]
                
                # Remove the input product if it's in the recommendations
                top_indices = top_indices[top_indices != encoded_product_id][:n_recommendations]
                
                recommendations = []
                for idx in top_indices:
                    recommended_original_id = self.product_id_encoder.inverse_transform([idx])[0]
                    
                    # Find product details
                    if self.data is not None:
                        product_matches = self.data[self.data['product_id'] == recommended_original_id]
                        if len(product_matches) > 0:
                            recommended_product_details = product_matches.iloc[0]
                        else:
                            continue
                    elif self.popularity_model is not None:
                        product_matches = self.popularity_model[self.popularity_model['product_id'] == recommended_original_id]
                        if len(product_matches) > 0:
                            recommended_product_details = product_matches.iloc[0]
                        else:
                            continue
                    else:
                        continue
                    
                    recommendation_info = {
                        'product_id': recommended_original_id,
                        'product_name': recommended_product_details['product_name'],
                        'category': recommended_product_details['category'],
                        'rating': recommended_product_details['rating'],
                        'rating_count': recommended_product_details['rating_count'],
                        'img_link': recommended_product_details['img_link'],
                        'product_link': recommended_product_details['product_link'],
                        'similarity': float(similarities[idx]),
                        'source': 'dmf'
                    }
                    recommendations.append(recommendation_info)
                
                return recommendations
        except Exception as e:
            print(f"Error in DMF recommendations: {e}")
            return []
    
    def get_popularity_recommendations(self, category=None, n_recommendations=5):
        """Get recommendations using popularity model"""
        if self.popularity_model is None:
            return []
            
        # Filter data by category if specified
        if category and category != "All":
            filtered_data = self.popularity_model[self.popularity_model['category'] == category]
        else:
            filtered_data = self.popularity_model
            
        # Get top n products
        top_products = filtered_data.head(n_recommendations)
        
        recommendations = []
        for _, product_details in top_products.iterrows():
            try:
                recommendation_info = {
                    'product_id': product_details['product_id'],
                    'product_name': product_details['product_name'],
                    'category': product_details['category'],
                    'rating': product_details['rating'],
                    'rating_count': product_details['rating_count'],
                    'img_link': product_details['img_link'],
                    'product_link': product_details['product_link'],
                    'similarity': float(product_details.get('popularity_score', 0) / 1000),  # Normalize score
                    'source': 'popularity'
                }
                recommendations.append(recommendation_info)
            except Exception as e:
                print(f"Error adding popularity recommendation: {e}")
                continue
        
        return recommendations
    
    def get_personalized_recommendations(self, search_keywords, n_recommendations=5):
        """Get personalized recommendations based on keywords"""
        if self.keywords_data is None:
            return self.get_popularity_recommendations(n_recommendations=n_recommendations)
            
        try:
            # Convert keywords to lowercase
            search_keywords = search_keywords.lower().split()
            
            # Find product_ids containing keywords
            matching_product_ids = set()
            for keyword in search_keywords:
                matches = self.keywords_data[self.keywords_data['Keyword'].str.lower().str.contains(keyword, na=False)]
                matching_product_ids.update(matches['Product_ID'].tolist())
            
            # If no results, return popularity recommendations
            if not matching_product_ids:
                return self.get_popularity_recommendations(n_recommendations=n_recommendations)
            
            # Filter and sort by popularity
            if self.popularity_model is not None:
                data_source = self.popularity_model
            else:
                data_source = self.data
                
            matched_products = data_source[data_source['product_id'].isin(matching_product_ids)]
            
            if 'popularity_score' in matched_products.columns:
                matched_products = matched_products.sort_values('popularity_score', ascending=False)
            
            # Get top n products
            top_products = matched_products.head(n_recommendations)
            
            recommendations = []
            for _, product_details in top_products.iterrows():
                similarity_score = float(product_details.get('popularity_score', 0) / 1000)
                
                recommendation_info = {
                    'product_id': product_details['product_id'],
                    'product_name': product_details['product_name'],
                    'category': product_details['category'],
                    'rating': product_details['rating'],
                    'rating_count': product_details['rating_count'],
                    'img_link': product_details['img_link'],
                    'product_link': product_details['product_link'],
                    'similarity': similarity_score,
                    'source': 'personalized'
                }
                recommendations.append(recommendation_info)
            
            return recommendations
        except Exception as e:
            print(f"Error in personalized recommendations: {e}")
            return self.get_popularity_recommendations(n_recommendations=n_recommendations)
    
    def get_hybrid_recommendations(self, product_id=None, category=None, search_keywords=None, n_recommendations=5):
        """
        Get hybrid recommendations combining DMF and popularity
        
        Args:
            product_id: ID of the input product (for DMF)
            category: Product category (for filtering popularity)
            search_keywords: Search keywords for personalization
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # If search keywords provided, use personalized recommendations
        if search_keywords:
            return self.get_personalized_recommendations(search_keywords, n_recommendations)
        
        # If product_id provided, use DMF combined with popularity
        if product_id:
            dmf_recs = self.get_dmf_recommendations(product_id, n_recommendations=n_recommendations*2)
            pop_recs = self.get_popularity_recommendations(category, n_recommendations=n_recommendations*2)
            
            # Combine recommendations
            all_recs = dmf_recs + pop_recs
            
            # Create a dictionary to store the highest similarity score for each product
            product_scores = {}
            for rec in all_recs:
                rec_product_id = rec['product_id']
                
                # Calculate weighted similarity score based on source
                weight = self.dmf_weight if rec['source'] == 'dmf' else self.pop_weight
                
                weighted_score = rec['similarity'] * weight
                
                if rec_product_id not in product_scores or weighted_score > product_scores[rec_product_id]['score']:
                    product_scores[rec_product_id] = {
                        'rec': rec,
                        'score': weighted_score
                    }
            
            # Sort products by score and take top n
            sorted_products = sorted(product_scores.values(), key=lambda x: x['score'], reverse=True)
            recommendations = [item['rec'] for item in sorted_products[:n_recommendations]]
        else:
            # If no product_id, use only popularity model
            recommendations = self.get_popularity_recommendations(category, n_recommendations)
        
        return recommendations