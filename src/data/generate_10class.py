"""
Enhanced Synthetic Data Generation for 10-Class Medical Text Classification
Includes new Sx (Symptoms) class for comprehensive medical text coverage
"""

import pandas as pd
import random
import yaml
import os
from typing import List, Tuple, Dict
import numpy as np


class MedicalDataGenerator:
    """
    Enhanced data generator for 10-class medical text classification
    Includes all original classes plus new Sx (Symptoms) class
    """
    
    def __init__(self, config_path: str = "config/classes.yaml"):
        """Initialize with class configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.num_classes = self.config['num_classes']
        self.class_labels = self.config['class_labels']
        self.class_configs = self.config['class_configs']
        self.data_config = self.config['data_generation']
        
        random.seed(self.data_config['random_seed'])
        np.random.seed(self.data_config['random_seed'])
        
    def generate_rx_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic medication data"""
        medications = [
            ("metformin", "for type 2 diabetes management"),
            ("lisinopril", "ACE inhibitor for hypertension"),
            ("atorvastatin", "statin for cholesterol management"),
            ("levothyroxine", "thyroid hormone replacement"),
            ("amlodipine", "calcium channel blocker for blood pressure"),
            ("metoprolol", "beta blocker for heart conditions"),
            ("omeprazole", "proton pump inhibitor for acid reflux"),
            ("sertraline", "SSRI antidepressant medication"),
            ("gabapentin", "for neuropathic pain management"),
            ("furosemide", "diuretic for fluid retention"),
            ("insulin", "for diabetes blood sugar control"),
            ("warfarin", "anticoagulant blood thinner"),
            ("aspirin", "for cardiovascular protection"),
            ("prednisone", "corticosteroid for inflammation"),
            ("albuterol", "bronchodilator for asthma")
        ]
        
        dosages = ["10mg", "20mg", "25mg", "50mg", "100mg", "500mg", "1000mg"]
        frequencies = ["once daily", "twice daily", "three times daily", "as needed", "every 8 hours"]
        
        templates = [
            "Patient prescribed {drug} {dosage} {frequency} {description}",
            "Started on {drug} {dosage} {frequency}, {description}",
            "Continue {drug} therapy {dosage} {frequency}, {description}",
            "Medication: {drug} {dosage} {frequency} - {description}",
            "Rx: {drug} {dosage} {frequency} ({description})",
            "Taking {drug} {dosage} {frequency}, {description}",
            "Dispensed {drug} {dosage} {frequency}, {description}",
            "Refill {drug} prescription {dosage} {frequency}, {description}"
        ]
        
        data = []
        for _ in range(n_samples):
            drug, desc = random.choice(medications)
            dosage = random.choice(dosages)
            frequency = random.choice(frequencies)
            template = random.choice(templates)
            text = template.format(drug=drug, dosage=dosage, frequency=frequency, description=desc)
            data.append((text, "Rx"))
        
        return data
    
    def generate_lx_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic laboratory test data"""
        lab_tests = [
            ("hemoglobin A1c", "7.2%", "measures average blood glucose"),
            ("complete blood count", "normal", "evaluates blood cell types"),
            ("basic metabolic panel", "within limits", "checks electrolytes and kidney function"),
            ("lipid panel", "elevated LDL 140", "measures cholesterol levels"),
            ("thyroid stimulating hormone", "2.5 mIU/L", "evaluates thyroid function"),
            ("creatinine", "1.1 mg/dL", "measures kidney function"),
            ("liver function tests", "ALT 45", "evaluates liver enzymes"),
            ("urinalysis", "protein trace", "examines urine composition"),
            ("prostate specific antigen", "2.8 ng/mL", "prostate cancer screening"),
            ("cardiac enzymes", "troponin elevated", "measures heart muscle damage")
        ]
        
        templates = [
            "Lab result: {test} {result}, {description}",
            "Ordered {test} - {result}, {description}",
            "{test} shows {result}, {description}",
            "Laboratory: {test} {result} ({description})",
            "Test results: {test} {result}, {description}",
            "{test} levels {result}, {description}"
        ]
        
        data = []
        for _ in range(n_samples):
            test, result, desc = random.choice(lab_tests)
            template = random.choice(templates)
            text = template.format(test=test, result=result, description=desc)
            data.append((text, "Lx"))
        
        return data
    
    def generate_dx_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic diagnosis data"""
        diagnoses = [
            ("type 2 diabetes mellitus", "chronic metabolic disorder"),
            ("essential hypertension", "high blood pressure condition"),
            ("hyperlipidemia", "elevated cholesterol levels"),
            ("coronary artery disease", "narrowing of heart vessels"),
            ("chronic kidney disease", "progressive kidney function decline"),
            ("major depressive disorder", "persistent mood disorder"),
            ("generalized anxiety disorder", "chronic anxiety condition"),
            ("osteoarthritis", "joint cartilage degeneration"),
            ("chronic obstructive pulmonary disease", "lung airway obstruction"),
            ("gastroesophageal reflux disease", "acid reflux condition")
        ]
        
        templates = [
            "Patient diagnosed with {condition}, {description}",
            "Diagnosis: {condition} - {description}",
            "Clinical impression: {condition}, {description}",
            "Assessment: {condition}, {description}",
            "Primary diagnosis {condition}, {description}",
            "Confirmed diagnosis of {condition}, {description}"
        ]
        
        data = []
        for _ in range(n_samples):
            condition, desc = random.choice(diagnoses)
            template = random.choice(templates)
            text = template.format(condition=condition, description=desc)
            data.append((text, "Dx"))
        
        return data
    
    def generate_px_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic procedure data"""
        procedures = [
            ("cardiac catheterization", "diagnostic heart procedure"),
            ("colonoscopy", "colon cancer screening procedure"),
            ("arthroscopic knee surgery", "minimally invasive joint procedure"),
            ("cataract extraction", "eye lens replacement surgery"),
            ("endoscopic procedure", "internal examination via scope"),
            ("blood pressure monitoring", "continuous vital sign assessment"),
            ("physical therapy", "rehabilitation exercise program"),
            ("wound care", "chronic wound management"),
            ("vaccination administration", "preventive immunization"),
            ("routine follow-up", "scheduled monitoring visit")
        ]
        
        templates = [
            "Patient underwent {procedure}, {description}",
            "Procedure: {procedure} - {description}",
            "Performed {procedure}, {description}",
            "Completed {procedure}, {description}",
            "Scheduled for {procedure}, {description}",
            "{procedure} performed successfully, {description}"
        ]
        
        data = []
        for _ in range(n_samples):
            procedure, desc = random.choice(procedures)
            template = random.choice(templates)
            text = template.format(procedure=procedure, description=desc)
            data.append((text, "Px"))
        
        return data
    
    def generate_sx_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic symptoms data - NEW CLASS!"""
        symptom_categories = {
            'pain': [
                ("chest pain", "sharp, radiating to left arm"),
                ("abdominal pain", "cramping, lower right quadrant"),
                ("headache", "throbbing, bilateral temporal"),
                ("back pain", "chronic, lower lumbar region"),
                ("joint pain", "morning stiffness, multiple joints"),
                ("muscle pain", "generalized myalgia")
            ],
            'respiratory': [
                ("shortness of breath", "worsening with exertion"),
                ("persistent cough", "dry, non-productive"),
                ("wheezing", "expiratory wheeze on exam"),
                ("chest tightness", "pressure sensation"),
                ("difficulty breathing", "especially at night")
            ],
            'gastrointestinal': [
                ("nausea", "persistent, worse after eating"),
                ("vomiting", "multiple episodes daily"),
                ("diarrhea", "watery, frequent stools"),
                ("constipation", "less than 3 bowel movements per week"),
                ("abdominal bloating", "fullness after meals"),
                ("heartburn", "burning sensation, chest area")
            ],
            'neurological': [
                ("dizziness", "spinning sensation when standing"),
                ("numbness", "bilateral hands and feet"),
                ("tingling", "pins and needles sensation"),
                ("weakness", "progressive muscle weakness"),
                ("confusion", "difficulty concentrating"),
                ("memory loss", "short-term memory problems")
            ],
            'psychiatric': [
                ("anxiety", "persistent worry and nervousness"),
                ("depression", "persistent low mood"),
                ("mood changes", "irritability and emotional lability"),
                ("sleep disturbance", "difficulty falling asleep"),
                ("fatigue", "persistent tiredness despite rest"),
                ("panic attacks", "episodes of intense fear")
            ],
            'constitutional': [
                ("fever", "temperature elevation to 101.5°F"),
                ("chills", "shaking and feeling cold"),
                ("night sweats", "drenching perspiration"),
                ("weight loss", "unintentional 15 lb decrease"),
                ("weight gain", "10 lb increase over 2 months"),
                ("appetite changes", "decreased food intake")
            ]
        }
        
        templates = [
            "Patient reports {symptom}, {description}",
            "Chief complaint: {symptom}, {description}",
            "Patient complains of {symptom}, {description}",
            "Presenting symptom: {symptom}, {description}",
            "Patient experiences {symptom}, {description}",
            "Symptom: {symptom} - {description}",
            "Patient describes {symptom}, {description}",
            "Ongoing {symptom}, {description}"
        ]
        
        data = []
        for _ in range(n_samples):
            category = random.choice(list(symptom_categories.keys()))
            symptom, desc = random.choice(symptom_categories[category])
            template = random.choice(templates)
            text = template.format(symptom=symptom, description=desc)
            data.append((text, "Sx"))
        
        return data
    
    def generate_other_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic non-clinical data"""
        other_content = [
            ("insurance verification completed", "administrative task"),
            ("appointment scheduling", "patient coordination"),
            ("contact information updated", "demographic update"),
            ("billing inquiry", "financial matter"),
            ("referral authorization", "administrative approval"),
            ("medical records request", "documentation request"),
            ("insurance pre-authorization", "coverage verification"),
            ("patient portal access", "technical support"),
            ("pharmacy benefits check", "insurance coverage"),
            ("transportation assistance", "social services")
        ]
        
        templates = [
            "{content} - {description}",
            "Administrative: {content}, {description}",
            "Note: {content}, {description}",
            "{content} ({description})",
            "Non-clinical: {content}, {description}"
        ]
        
        data = []
        for _ in range(n_samples):
            content, desc = random.choice(other_content)
            template = random.choice(templates)
            text = template.format(content=content, description=desc)
            data.append((text, "Other"))
        
        return data
    
    def generate_vitals_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic vital signs data"""
        vital_measurements = [
            ("blood pressure", "135/85 mmHg", "slightly elevated"),
            ("heart rate", "72 bpm", "regular rhythm"),
            ("temperature", "98.6°F", "normal body temperature"),
            ("respiratory rate", "16 breaths/min", "normal breathing"),
            ("oxygen saturation", "98%", "adequate oxygenation"),
            ("height", "5'8\"", "adult height measurement"),
            ("weight", "180 lbs", "current body weight"),
            ("BMI", "27.3", "slightly overweight category"),
            ("pain scale", "4/10", "moderate discomfort level"),
            ("Glasgow coma scale", "15/15", "normal neurological status")
        ]
        
        templates = [
            "Vital signs: {vital} {measurement}, {description}",
            "{vital}: {measurement} ({description})",
            "Measured {vital} {measurement}, {description}",
            "Vitals show {vital} {measurement}, {description}",
            "{vital} recorded as {measurement}, {description}"
        ]
        
        data = []
        for _ in range(n_samples):
            vital, measurement, desc = random.choice(vital_measurements)
            template = random.choice(templates)
            text = template.format(vital=vital, measurement=measurement, description=desc)
            data.append((text, "Vitals"))
        
        return data
    
    def generate_fhx_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic family history data"""
        family_relations = ["mother", "father", "sister", "brother", "grandmother", "grandfather", "aunt", "uncle"]
        conditions = ["diabetes", "hypertension", "heart disease", "cancer", "stroke", "dementia", "depression", "arthritis"]
        
        templates = [
            "Family history: {relation} with {condition}",
            "Strong family history of {condition} on {relation}'s side",
            "{relation} has {condition}",
            "Family history significant for {condition} in {relation}",
            "Positive family history: {relation} diagnosed with {condition}",
            "Hereditary risk: {condition} in {relation}"
        ]
        
        data = []
        for _ in range(n_samples):
            relation = random.choice(family_relations)
            condition = random.choice(conditions)
            template = random.choice(templates)
            text = template.format(relation=relation, condition=condition)
            data.append((text, "FHx"))
        
        return data
    
    def generate_sdoh_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic social determinants data"""
        sdoh_factors = [
            ("housing instability", "affecting medication storage"),
            ("transportation barriers", "difficulty accessing care"),
            ("food insecurity", "impacting nutritional status"),
            ("financial stress", "affecting medication compliance"),
            ("social isolation", "limited support system"),
            ("language barriers", "need for interpreter services"),
            ("educational limitations", "health literacy concerns"),
            ("unemployment", "affecting insurance coverage"),
            ("elder care responsibilities", "caregiver burden"),
            ("neighborhood safety", "limiting outdoor activities")
        ]
        
        templates = [
            "Social determinant: {factor}, {impact}",
            "SDOH: {factor} - {impact}",
            "Social factor: {factor}, {impact}",
            "Patient reports {factor}, {impact}",
            "Social issue: {factor}, {impact}"
        ]
        
        data = []
        for _ in range(n_samples):
            factor, impact = random.choice(sdoh_factors)
            template = random.choice(templates)
            text = template.format(factor=factor, impact=impact)
            data.append((text, "SDOH"))
        
        return data
    
    def generate_tobacco_data(self, n_samples: int) -> List[Tuple[str, str]]:
        """Generate synthetic tobacco use data"""
        tobacco_patterns = [
            ("current smoker", "1 pack per day for 15 years"),
            ("former smoker", "quit 5 years ago, 20 pack-year history"),
            ("never smoker", "denies tobacco use"),
            ("occasional smoker", "social smoking on weekends"),
            ("trying to quit", "using nicotine replacement therapy"),
            ("heavy smoker", "2 packs per day for 25 years"),
            ("vaping", "e-cigarette use daily"),
            ("chewing tobacco", "smokeless tobacco use"),
            ("pipe smoker", "occasional pipe tobacco"),
            ("quit recently", "stopped smoking 6 months ago")
        ]
        
        templates = [
            "Tobacco history: {pattern}, {details}",
            "Smoking status: {pattern}, {details}",
            "Patient is a {pattern}, {details}",
            "Tobacco use: {pattern} - {details}",
            "{pattern}, {details}"
        ]
        
        data = []
        for _ in range(n_samples):
            pattern, details = random.choice(tobacco_patterns)
            template = random.choice(templates)
            text = template.format(pattern=pattern, details=details)
            data.append((text, "Tobacco"))
        
        return data
    
    def generate_all_data(self) -> pd.DataFrame:
        """Generate complete dataset with all 10 classes"""
        samples_per_class = self.data_config['samples_per_class']
        
        print("Generating synthetic data for 10-class medical text classification...")
        print(f"Samples per class: {samples_per_class}")
        
        all_data = []
        
        # Generate data for each class
        generators = {
            'Rx': self.generate_rx_data,
            'Lx': self.generate_lx_data,
            'Dx': self.generate_dx_data,
            'Px': self.generate_px_data,
            'Sx': self.generate_sx_data,  # NEW!
            'Other': self.generate_other_data,
            'Vitals': self.generate_vitals_data,
            'FHx': self.generate_fhx_data,
            'SDOH': self.generate_sdoh_data,
            'Tobacco': self.generate_tobacco_data
        }
        
        for class_name, generator in generators.items():
            print(f"Generating {class_name} data...")
            class_data = generator(samples_per_class)
            all_data.extend(class_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=['text', 'label'])
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=self.data_config['random_seed']).reset_index(drop=True)
        
        # Add numeric labels
        label_to_id = self.config['label_to_id']
        df['label_id'] = df['label'].map(label_to_id)
        
        print(f"\nGenerated {len(df)} total samples")
        print(f"Class distribution:")
        print(df['label'].value_counts().sort_index())
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_path: str = "data/synthetic_training_data_10class.csv"):
        """Save generated data to CSV file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nData saved to: {output_path}")
        
        # Save summary statistics
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("10-Class Medical Text Classification Dataset Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Number of classes: {self.num_classes}\n\n")
            f.write("Class distribution:\n")
            for label, count in df['label'].value_counts().sort_index().items():
                f.write(f"{label}: {count} samples\n")
            f.write(f"\nClass labels mapping:\n")
            for id, label in self.class_labels.items():
                f.write(f"{id}: {label}\n")
        
        print(f"Summary saved to: {summary_path}")


def main():
    """Main function to generate and save synthetic data"""
    generator = MedicalDataGenerator()
    df = generator.generate_all_data()
    generator.save_data(df)
    
    # Display sample data
    print("\nSample data:")
    print("="*80)
    for label in generator.class_labels.values():
        sample = df[df['label'] == label].iloc[0]
        print(f"{label}: {sample['text']}")
    print("="*80)


if __name__ == "__main__":
    main()
